package hw1;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.*;


public class MutualFriendRecommend {

    
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        Job job = new Job(conf, "MutualFriendRecommend");
        job.setJarByClass(MutualFriendRecommend.class);

        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(FriendsCountWritable.class);

        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        //FileSystem outFs = new Path(args[1]).getFileSystem(conf);
        //outFs.delete(new Path(args[1]), true);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.waitForCompletion(true);
    }

    // Class to define data type for storing recommended friend and mutual friend
    static public class FriendsCountWritable implements Writable {
        public Long userID;
        public Long mutualFriends;

        public FriendsCountWritable(Long userID, Long mutualFriends) {
            this.userID = userID;
            this.mutualFriends = mutualFriends;
        }

        public FriendsCountWritable() {
            this(-1L, -1L);
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeLong(userID);
            out.writeLong(mutualFriends);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            userID = in.readLong();
            mutualFriends = in.readLong();
        }

        @Override
        public String toString() {
            return " toUser: "
                    + Long.toString(userID) + " mutualFriends: " + Long.toString(mutualFriends);
        }
    }

    // Map 
    public static class Map extends Mapper<LongWritable, Text, LongWritable, FriendsCountWritable> {
        private Text word = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            String line[] = value.toString().split("\t");  
            Long user = Long.parseLong(line[0]);

            if (line.length == 2) {
                // User has friends (Not an empty list)
                
                String[] friends_of_user = line[1].toString().split(",");
                for (int i = 0; i < friends_of_user.length; i++) {
                    // Adding entry for user and its friend (To check if recommended friend is already friend of user)
                    context.write(new LongWritable(user), new FriendsCountWritable(Long.parseLong(friends_of_user[i]), -1L)); 
                    for (int j = i + 1; j < friends_of_user.length; j++) {
                        // Storing mutual friends i.e. If 1 has friends 2, 3; then storing 2 as recommended friend of 3 via mutual friend 1 and vica versa
                        context.write(new LongWritable(Long.parseLong(friends_of_user[i])), new FriendsCountWritable(Long.parseLong((friends_of_user[j])), user));
                        context.write(new LongWritable(Long.parseLong(friends_of_user[j])), new FriendsCountWritable(Long.parseLong((friends_of_user[i])), user));
                    }
                }
            }
        }
    }

    // Reduce
    public static class Reduce extends Reducer<LongWritable, FriendsCountWritable, LongWritable, Text> {
        @Override
        public void reduce(LongWritable key, Iterable<FriendsCountWritable> values, Context context)
                throws IOException, InterruptedException {
                    
            // HashMap to store mutual friends record: key is the recommended friend, and value is the list of mutual friends
            final java.util.Map<Long, List<Long>> mutualFriendsMap = new HashMap<Long, List<Long>>();

            for (FriendsCountWritable val : values) {
                Boolean isFriend = false;
                if (val.mutualFriends == -1)
                {
                    isFriend = true;
                }
                final Long user = val.userID;
                final Long mutualFriend = val.mutualFriends;

                if (mutualFriendsMap.containsKey(user)) {
                    if (isFriend) {
                        mutualFriendsMap.put(user, null);
                    } else if (mutualFriendsMap.get(user) != null) {
                        mutualFriendsMap.get(user).add(mutualFriend);
                    }
                } else {
                    if (!isFriend) {
                        mutualFriendsMap.put(user, new ArrayList<Long>() {
                            {
                                add(mutualFriend);
                            }
                        });
                    } else {
                        mutualFriendsMap.put(user, null);
                    }
                }
            }

            // Using a TreeMap to sort the mutualFriendsMap containing recommendations:
            java.util.SortedMap<Long, List<Long>> sortMutualFriends = new TreeMap<Long, List<Long>>(new Comparator<Long>() {
                @Override
                public int compare(Long key1, Long key2) {
                    Integer v1 = mutualFriendsMap.get(key1).size();
                    Integer v2 = mutualFriendsMap.get(key2).size();
                    // Sorting mutual friends in descending order
                    if (v1 > v2) {
                        return -1;
                    } else if (v1.equals(v2) && key1 < key2) {
                        // If number of mutual friends are same, sorting in terms of ascending value of userID
                        return -1;
                    } else {
                        return 1;
                    }
                }
            });

            // Sorting the contents of HashMap by using sortMutualFriends:
            for (java.util.Map.Entry<Long, List<Long>> entry : mutualFriendsMap.entrySet()) {
                if (entry.getValue() != null) {
                    sortMutualFriends.put(entry.getKey(), entry.getValue());
                }
            }

            // Output generation:
            Integer i = 0;
            String output = "";
            for (java.util.Map.Entry<Long, List<Long>> entry : sortMutualFriends.entrySet()) {
                if (i == 0) {
                    output = entry.getKey().toString() + " (" + entry.getValue().size() + ": " + entry.getValue() + ")";
                } else {
                    output += "," + entry.getKey().toString() + " (" + entry.getValue().size() + ": " + entry.getValue() + ")";
                }
                ++i;
            }
            context.write(key, new Text(output));   // Displays user and corresponding recommendations
        }
    }

}
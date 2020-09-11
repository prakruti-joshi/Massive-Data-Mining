
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.lang.Math;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class Kmeans extends Configured implements Tool {
	public static void main(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));
		int exitCode = 0;
        int MAX_ITR = 20;
		for(int i = 0; i < MAX_ITR; i++) {
			args[2] = Integer.toString(i);
			exitCode = ToolRunner.run(new Configuration(), new Kmeans(), args);
		}
		System.exit(exitCode);
	}

	@Override
	public int run(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));
		Job job = new Job(getConf(), "Kmeans");

		int curr = Integer.parseInt(args[2]);
		int next = curr + 1;
		Configuration conf = job.getConfiguration();
		// Change paths over here
		if(curr == 0) {
			conf.set("inputDir", "/Users/prakrutijoshi/Desktop/Rutgers_Sem2/MDM/Homeworks/homework2/data/hw2-q4-kmeans/c2.txt");
		}
		else {
			conf.set("inputDir", "/Users/prakrutijoshi/Desktop/Rutgers_Sem2/MDM/Homeworks/homework2/res_" + curr + ".txt");
		}
		conf.set("outputDir", "/Users/prakrutijoshi/Desktop/Rutgers_Sem2/MDM/Homeworks/homework2/res_" + next + ".txt");
		conf.set("costDir", "/Users/prakrutijoshi/Desktop/Rutgers_Sem2/MDM/Homeworks/homework2/cost.txt");

		
		job.setJarByClass(Kmeans.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		job.setMapperClass(Map.class);
		job.setReducerClass(Reduce.class);

		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(Text.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.addInputPath(job, new Path(args[0]));	
		FileOutputFormat.setOutputPath(job, new Path(args[1] + curr));

		job.waitForCompletion(true);

		return 0;
	}

	private static double cost;
	
	public static class Map extends Mapper<LongWritable, Text, IntWritable, Text> {
        
		List<Double[]> centroids = new ArrayList<Double[]>();  // List of all the centroids

		protected void setup(Context context) throws IOException, InterruptedException {
			cost = 0;    // Resetting cost for each iteration

			// Reading file from input directory
			String fileName = context.getConfiguration().get("inputDir");
			File file = new File(fileName);
			BufferedReader reader = null;
			try {
				reader = new BufferedReader(new FileReader(file));
				String temp = null;
                // Reading file line by line: each line contains the centroid values of previous iteration
				while ((temp = reader.readLine()) != null) {
					String line = temp;
					String[] centroid_str = line.trim().split("\\s");
                    Double[] centroid_val = new Double[centroid_str.length];
                    for(int i = 0; i < centroid_str.length; i++) {
                         centroid_val[i] = Double.parseDouble(centroid_str[i]);
                    }
					centroids.add(centroid_val);
				}
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				if (reader != null) {
					try {
						reader.close();
					} catch (IOException e1) {
					}
				}
			}
		}
		
		protected void cleanup(Context context) throws IOException, InterruptedException {
			String fileName = context.getConfiguration().get("costDir");
            // Writes the cost for each iteration into the cost directory
			try{
				FileWriter writer = new FileWriter(fileName, true);
				writer.write(cost + "\n");
				writer.close();
			} catch (IOException e)
			{
				e.printStackTrace();
			}
		}

		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String str_val = value.toString();
			String[] str_arr = str_val.trim().split("\\s");
            Double[] point = new Double[str_arr.length];
            for(int i = 0; i < str_arr.length; i++) {
                point[i] = Double.parseDouble(str_arr[i]);
            }
            // Assigns point to nearest cluster by computing Euclidean distance with each cluster
			double min_dist = Double.MAX_VALUE;
			int index = -1;
			for(int i = 0; i < centroids.size(); i++) {
				double dist = calculate_euclid_dist(point, centroids.get(i));
				if(dist < min_dist) {
					min_dist = dist;
					index = i;
				}
			}
            // Writes the output
            // Key: index of nearest cluster , value : point
			context.write(new IntWritable(index), new Text(str_val));
            // Adds the distance to the cost
			cost += min_dist;
		}
	}
	
    // Euclidean distance:
	public static double calculate_euclid_dist(Double[] x, Double[] y) {
		double res = 0;
        int d = x.length;
		for(int i = 0; i < d; i++) {
			res += Math.pow((x[i] - y[i]),2);
		}
		return res;
	}

	public static class Reduce extends Reducer<IntWritable, Text, Text, Text> {

		@Override
		public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException { 
            int num_features = 58;  	  
			double[] centroid_sum = new double[num_features];
			int c_count = 0;

            // Iterates over value of same key and calculates the sum and count of points belonging to same cluster 
			for(Text value : values) {
				c_count++;
				String str_val = value.toString();
				String[] point = str_val.trim().split("\\s");
				for(int i = 0; i < num_features; i++) {
					centroid_sum[i] += Double.parseDouble(point[i]);
				}
			}

            // Calculates the new cluster centroid
			double[] centroid_avg = new double[num_features];
			for(int i = 0; i < num_features; i++) {
				centroid_avg[i] = centroid_sum[i] / c_count;
			}
			String res = "";
			for(int i = 0; i < num_features; i++) {
				res = res + " " + Double.toString(centroid_avg[i]);
			}
			String result = res.substring(1);  // To remove the first space
            // Writes the new output cluster values
			context.write(new Text(result), new Text(""));

			String fileName = context.getConfiguration().get("outputDir");
			try{
				FileWriter writer = new FileWriter(fileName, true);
				writer.write(result + "\n");
				writer.close();
			} catch (IOException e)
			{
				e.printStackTrace();
			}

		}
	}
}

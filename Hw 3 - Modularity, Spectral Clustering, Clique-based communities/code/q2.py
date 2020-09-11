from scipy import linalg
import numpy as np

L = np.array([[4, -1, -1, -1, 0, 0, -1, 0],
		[-1, 3, -1, -1, 0, 0, 0, 0],
		[-1, -1, 3, -1, 0, 0, 0, 0],
		[-1, -1, -1, 3, 0, 0, 0, 0],
		[0, 0, 0, 0, 2, -1, -1, 0],
		[0, 0, 0, 0, -1, 2, -1, 0],
		[-1, 0, 0, 0, -1, -1, 4, -1],
		[0, 0, 0, 0, 0, 0, -1, 1]])

node_mapping = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H'}


evals, evecs = linalg.eigh(L)
print("\nEigen values:")
# print(evals)
l = []
for v in evals:
	l.append(v)
print(l)
	

print("\nEigen vectors: ")
print(evecs, "\n")


lambda2 = evals[1]
x = evecs[:,1]
print(lambda2)
print(x)

# Community 1
print("Community 1")
for n in np.argwhere(x>0):
	print(node_mapping[n[0]])

# Community 2 
print("\n Community 2")
for n in np.argwhere(x<0):
	print(node_mapping[n[0]])
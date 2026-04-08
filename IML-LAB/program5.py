import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data=np.array([
    [2,10],
    [2,5],
    [8,4],
    [5,8],
    [7,5],
    [6,4],
    [1,2],
    [4,9]
])
point_names=['a1','a2','a3','b1','b2','b3','c1','c2']
k=3
Kmeans=KMeans(n_clusters=k,random_state=42,n_init=10)
Kmeans.fit(data)
labels=Kmeans.labels_
centroids=Kmeans.cluster_centers_

print("--- k-Means numerical rseult ---")
print(f"{'point':<6} | {'coordinates':<10} | {'cluster assignment'}")
print("-"*40)
for i in range(len(data)):
  print(f"{point_names[i]:<6} | {str(data[i]):<10} | cluster {labels[i]}")

print("\n--- final centroids ---")
for j, center in enumerate(centroids):
  print(f"cluster{j} centroid: {center}")

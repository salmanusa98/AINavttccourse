import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Sample data points
X = np.array([[1, 2], [2, 3], [3, 4], [5, 7], [6, 8], [7, 9]])

# Run KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plotting the data points and centroids
plt.figure(figsize=(8, 6))

# Plot the data points colored by their cluster
for i in range(len(X)):
    plt.scatter(X[i, 0], X[i, 1], c=['orange', 'purple'][labels[i]], label=f'Point {i+1}')
    plt.text(X[i, 0]+0.1, X[i, 1]+0.1, f'P{i+1}', fontsize=9)

# Plot the centroids
for i, (cx, cy) in enumerate(centroids):
    plt.scatter(cx, cy, marker='X', s=200, c='red')
    plt.text(cx + 0.2, cy, f'Centroid {i}', fontsize=10, fontweight='bold', color='red')

plt.title('K-Means Clustering with Centroid Labels')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Sample customer data (Age, Annual Income)
data = {
    'Age': [25, 32, 47, 56, 23, 33, 45, 54, 23, 43],
    'Income': [25000, 35000, 50000, 60000, 22000, 38000, 48000, 56000, 24000, 45000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Standardize the data (Important for K-Means to perform well)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=0)  # We assume 3 clusters
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 4: Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['Age'], df['Income'], c=df['Cluster'], cmap='viridis', s=100, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200)
plt.title('Customer Segmentation (Age vs Income)')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.colorbar(label='Cluster')
plt.show()

# Step 5: Print the Cluster Centers and Customers in Each Cluster
print("Cluster Centers (Age, Income):\n", kmeans.cluster_centers_)
print("\nCustomer Data with Cluster Assignments:")
print(df)

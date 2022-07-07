from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Create sample data that shaped like a cresent moon
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# Plot the sample data
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1])
plt.tight_layout()

# Train the K-means clustering model and predict the output of the sample data
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)

# Train the agglomerative (hierarchical-tree) clustering model and predict the output of the sample data
ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')
y_ac = ac.fit_predict(X)

# Plot the fit for k-means and agglomerative clustering models
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1], # 1st cluster for k-means clustering model
            edgecolor='black',
            c='lightblue', marker='o', s=40, label='cluster 1')
ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1], # 2nd cluster for k-means clustering model
            edgecolor='black',
            c='red', marker='s', s=40, label='cluster 2')
ax1.set_title('K-means clustering')
ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c='lightblue', # 1st cluster for agglomerative clustering model
            edgecolor='black',
            marker='o', s=40, label='Cluster 1')
ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c='red', # 2nd cluster for agglomerative clustering model
            edgecolor='black',
            marker='s', s=40, label='Cluster 2')
ax2.set_title('Agglomerative clustering')
plt.legend()
plt.tight_layout()

# Train the DBSCAN (density-based) clustering model and predict the output of the sample data
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(X)

# Plot the fit for DBSCAN clustering models
plt.figure(3)
plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1], # 1st cluster for DBSCAN clustering model
            c='lightblue', marker='o', s=40,
            edgecolor='black', 
            label='Cluster 1')
plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1], # 2nd cluster for DBSCAN clustering model
            c='red', marker='s', s=40,
            edgecolor='black', 
            label='Cluster 2')
plt.legend()
plt.tight_layout()

plt.show()


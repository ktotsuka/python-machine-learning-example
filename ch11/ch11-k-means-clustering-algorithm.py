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

# Generate fake cluster data (note that the output "y" is the cluster number but is not used)
X, y = make_blobs(n_samples=150, # number of samples
                  n_features=2, # number of features
                  centers=3, # number of cluster centers (centroids)
                  cluster_std=0.5, # standard deviation of the clusters
                  shuffle=True, 
                  random_state=0)

# Plot the generated cluster data
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()

# Train the k-means model and predict output using the trained model
km = KMeans(n_clusters=3, # have to specify the number of clusters apriori
            init='random', # placement of the initial centroids.  Use 'k-means++' to use the K-means++ algorithm to select the initial centroids
            n_init=10, # number of time the k-means algorithm will be run with different centroid seeds.
                       # this could be helpful because bad centroid seeds lead to slow convergence.
            max_iter=300, # maximum number of iterations
            tol=1e-04, # error tolerance for declaring convergence
            random_state=0)
y_km = km.fit_predict(X)

# Plot the resulting clusters
plt.figure(2)
plt.scatter(X[y_km == 0, 0], # for data classified as cluster #0
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0], # for data classified as cluster #1
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Cluster 2')
plt.scatter(X[y_km == 2, 0], # for data classified as cluster #2
            X[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='Cluster 3')
plt.scatter(km.cluster_centers_[:, 0], # for plotting the centroid for each cluster
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='Centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()

# Computer SSE (sum of squared errors) for different number of clusters
distortions = []
for i in range(1, 11): # number of clusters to try
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_) # "km.inertia_" contains the SSE

# Plot the SSE vs. number of clusters to find the best number of clusters (the elbow method)
plt.figure(3)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()

plt.show()
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

# Train the k-means model and predict output using the trained model
km = KMeans(n_clusters=3, # have to specify the number of clusters apriori
            init='k-means++', # algorithm to select the initial centroids
            n_init=10, # number of time the k-means algorithm will be run with different centroid seeds.
                       # this could be helpful because bad centroid seeds lead to slow convergence.
            max_iter=300, # maximum number of iterations
            tol=1e-04, # error tolerance for declaring convergence
            random_state=0)
y_km = km.fit_predict(X)

# Compute the silhouette values for all samples
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')

# Plot the bar graph for the silhouette values for the 3 clusters
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
y_ax_lower, y_ax_upper = 0, 0
yticks = []
plt.figure(1)
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c] # silhouette values for this cluster
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)

# Plot the silhouette average
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()

# Train the k-means model and predict output using the trained model.  This time use 2 clusters to illustrate "bad" clustering
km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

# Plot the clusters with 2 centroids
plt.figure(2)
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50,
            c='lightgreen',
            edgecolor='black',
            marker='s',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50,
            c='orange',
            edgecolor='black',
            marker='o',
            label='Cluster 2')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=250, marker='*', c='red', label='Centroids')
plt.legend()
plt.grid()
plt.tight_layout()

# Compute the silhouette values for all samples
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')

# Plot the silhouette values for the 2 clusters
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
y_ax_lower, y_ax_upper = 0, 0
yticks = []
plt.figure(3)
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
# Plot the silhouette average
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()

plt.show()


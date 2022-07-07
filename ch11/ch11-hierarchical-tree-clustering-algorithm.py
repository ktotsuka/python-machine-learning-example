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

# Create a sample data
np.random.seed(123)
variables = ['X', 'Y', 'Z'] # feature names
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4'] # sample IDs
X = np.random.random_sample([5, 3])*10 # 5 samples with 3 features.  The values range from 0 to 10
df = pd.DataFrame(X, columns=variables, index=labels)

# Compute the raw distance between each samples and store them as a distance matrix
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                        columns=labels,
                        index=labels)

# Group together different clusters based on their similarities.  
# 'complete': similarities are determined based on the farthest samples
# output: it contains the cluster merging steps starting with each sample being its own cluster to all samples in one cluster
#         1st column: the first cluster to merge
#         2nd column: the second cluster to merge
#         3rd column: distance between the two clusters
#         4th column: number of samples in the merged cluster
row_clusters = linkage(pdist(df, metric='euclidean'), # condensed distances (row_dist contains redundunt entries)
                             method='complete')

# Run this in DEBUG CONSOLE to see row_clusters in a table format
pd.DataFrame(row_clusters, 
             columns=['row label 1', 'row label 2',
                      'distance', 'no. of items in clust.'],
             index=['cluster %d' % (i + 1) 
                    for i in range(row_clusters.shape[0])])

# Plot the dendrogram showing the cluster merging process
row_dendr = dendrogram(row_clusters, 
                       labels=labels)
plt.tight_layout()
plt.ylabel('Euclidean distance')

# Plot the dendrogram as a sub-plot
fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6]) # The dimensions [left, bottom, width, height] of the new Axes. All quantities are in fractions of figure width and height
row_dendr = dendrogram(row_clusters, orientation='left') # 'left' rotates the dendrogram
axd.set_xticks([]) # remove x axis tick marks
axd.set_yticks([]) # remove y axis tick marks
for i in axd.spines.values(): # Remove borders from dendrogram
    i.set_visible(False)

# Get the order of sample IDs displayed in the dendrogram.  The order is neededed for the labels in the heat map.  
# The order is reversed due to the orientation of the dendrogram and the heat map
df_rowclust = df.iloc[row_dendr['leaves'][::-1]] 

# Plot heat map.  The heat map represents the feature values with colors.
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])  # x-pos, y-pos, width, height for the heat map sub-plot
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r') # plot the heat map
fig.colorbar(cax) # add the translation bar from the feature value to color
axm.set_xticklabels([''] + list(df_rowclust.columns)) # x-axis label
axm.set_yticklabels([''] + list(df_rowclust.index)) # y-axis label

plt.show()

# Train the agglomerative clustering model (same as above, but using the model available in sklearn)
ac = AgglomerativeClustering(n_clusters=3, # number of clusters
                             affinity='euclidean', 
                             linkage='complete') # similarities are determined based on the farthest samples
labels = ac.fit_predict(X)

# Display the result of cluster group
# The result should be the same as before for 3 clusters.
# Result = [1, 0, 0, 2, 1] -> cluster 0: ID_1 and ID_2, cluster 1: ID_0 and ID_4, cluster 2: ID_3
print('Cluster labels: %s' % labels) 


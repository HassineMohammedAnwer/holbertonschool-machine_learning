#!/usr/bin/env python3
"""12. Agglomerative"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt
import numpy as np


def agglomerative(X, dist):
    """performs agglomerative clustering on a dataset:
    X is a numpy.ndarray of shape (n, d) containing the dataset
    dist is the maximum cophenetic distance for all clusters
    Performs agglomerative clustering with Ward linkage
    Displays the dendrogram with each cluster displayed in a different color
    The only imports you are allowed to use are:
    import scipy.cluster.hierarchy
    import matplotlib.pyplot as plt
    Returns: clss, a numpy.ndarray of shape (n,) containing the cluster
    __indices for each data point"""
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, 'ward')
    cluster_labels = scipy.cluster.hierarchy.fcluster(
        linkage_matrix, 
        t=dist, 
        criterion='distance'
    )
    plt.figure(figsize=(12, 8))
    dendrogram = scipy.cluster.hierarchy.dendrogram(
        linkage_matrix,
        color_threshold=dist,
        leaf_font_size=10,
    )
    
    plt.title('Hierarchical Clustering Dendrogram', fontsize=14)
    plt.xlabel('Data Points', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.axhline(y=dist, color='crimson', linestyle='--', 
                label=f'Distance Threshold = {dist}')
    plt.legend()
    num_clusters = len(np.unique(cluster_labels))
    print(f"Number of clusters formed: {num_clusters}")
    plt.tight_layout()
    plt.show()
    return cluster_labels

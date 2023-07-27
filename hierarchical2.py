import numpy as np
import pandas as pd

class HierarchicalClustering:
    def __init__(self, n_clusters, distance_metric='euclidean'):
        """
        Hierarchical Clustering initialization.

        Args:
            n_clusters (int): The desired number of clusters.
            distance_metric (str, optional): The distance metric to use. Defaults to 'euclidean'.
                Possible values: 'euclidean', 'manhattan', 'cosine'.
        """
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        
    def fit(self, X):
        """
        Perform hierarchical clustering.

        Args:
            X (DataFrame): Input data.

        """
        self.n_samples = X.shape[0]
        self.clusters = [[i] for i in range(self.n_samples)]
        self.history = []
        
        self.distances = self._compute_distance(X)
        
        while len(self.clusters) > self.n_clusters:
            i, j = self._find_closest_pair()
            self._merge_clusters(i, j)
            self.history.append((i, j))
            
    def get_distance(self, x1, x2):
        """
        Compute the distance between two samples.

        Args:
            x1 (Series): First sample.
            x2 (Series): Second sample.

        Returns:
            float: Distance between the samples.
        """
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(x1 - x2)
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'cosine':
            return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        else:
            raise ValueError('Invalid distance metric specified.')
    
    def _compute_distance(self, X):
        """
        Compute the pairwise distances between samples.

        Args:
            X (DataFrame): Input data.

        Returns:
            ndarray: Pairwise distances between samples.
        """
        distances = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                distances[i, j] = self.get_distance(X.iloc[i], X.iloc[j])
        return distances
        
    def _find_closest_pair(self):
        """
        Find the closest pair of clusters.

        Returns:
            tuple: Indices of the closest pair of clusters.
        """
        min_distance = np.inf
        closest_pair = None
        
        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                distance = self._calculate_cluster_distance(self.clusters[i], self.clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (i, j)
        return closest_pair
    
    def _calculate_cluster_distance(self, c1, c2):
        """
        Calculate the distance between two clusters.

        Args:
            c1 (list): Indices of samples in cluster 1.
            c2 (list): Indices of samples in cluster 2.

        Returns:
            float: Distance between the clusters.
        """
        distance = np.inf
        for i in c1:
            for j in c2:
                d = self.distances[min(i, j), max(i, j)]
                if d < distance:
                    distance = d
        return distance

    def _merge_clusters(self, i, j):
        """
        Merge two clusters.

        Args:
            i (int): Index of cluster 1.
            j (int): Index of cluster 2.
        """
        self.clusters[i] = self.clusters[i] + self.clusters[j]
        self.clusters.pop(j)
        
    def predict(self):
        """
        Assign cluster labels to samples.

        Returns:
            ndarray: Cluster labels for each sample.
        """
        labels = np.zeros(self.n_samples)
        
        for i, cluster in enumerate(self.clusters):
            for j in cluster:
                labels[j] = i
        return labels

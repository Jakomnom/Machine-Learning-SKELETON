import numpy as np
import pandas as pd

class HierarchicalClustering:
    def __init__(self, n_cluster):
        self.n_cluster = n_cluster
        
    def fit(self, X):
        self.n_samples = X.shape[0]
        self.clusters = [[i] for i in range (self.n_samples)]
        self.history = []
        
        self.distances = self._compute_distance(X)
        
        while len(self.clusters) > self.n_clusters:
            i, j = self._find_closest_pair()
            self._merge_clusters(i, j)
            self.history.append((i,j))
            
            
    def get_distance(self, x1, x2):
        return np.linalg.norm(x1, x2)
    
    def _compute_distance(self, X):
        distances = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(i+1, self.n_samples):
                distances[i, j] = self.get_distance(X.iloc[i], X.iloc[j])
            return distances
        
    def _find_closest_pair(self):
        
        min_distance = np.inf
        closest_pair = None
        
        for i in range(len(self.clusters)):
            for j in range(i+1, len(self.clusters)):
                distance = self.calculate_cluster_distance((self.clusters[i]), self.clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (i,j)
        return closest_pair
    
    def _calculate_cluster_distance(self, c1, c2):
        distance = np.inf
        for i in c1:
            for j in c2:
                d = self.distances[min(i,j), max(i,j)]
                if d < distance:
                    distance = d
        return distance

    def _merge_clusters(self, i, j):
        self.clusters[i] = self.clusters[i] + self.clusters[j]
        self.clusters.pop(j)
        
    def predict(self):
        
        labels = np.zeros(self.n_samples)
        
        for i, cluster in enumerate(self.clusters):
            for j in cluster:
                labels[j] = i
            return labels
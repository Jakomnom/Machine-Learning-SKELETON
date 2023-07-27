import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, k, max_iter):
        self.k = k
        self.max_iter = max_iter
        
    def _compute_centroid(self, assigned_centroid_dict):
        self.centroids = []
        for centroid, centroid_data in assigned_centroid_dict.items():
            centroid_mean = pd.DataFrame(centroid_data).mean().values
            self.centroids.append(centroid_mean)
        self.centroids = pd.DataFrame(self.centroids, columns=self.centroids[0].index)
            
    def fit(self, X):
        self.centroids = X.sample(self.k).reset_index(drop=True)
        
        for _ in range(self.max_iter):
            assigned_centroid_dict = self.predict(X)
            self._compute_centroid(assigned_centroid_dict)
            
    def predict(self, X):
        assigned_centroid_dict = {}
        
        for _, row_X in X.iterrows():
            assigned_centroid = np.argmin(np.linalg.norm(row_X - self.centroids, axis=1))
            
            if assigned_centroid not in assigned_centroid_dict:
                assigned_centroid_dict[assigned_centroid] = []
                
            assigned_centroid_dict[assigned_centroid].append(row_X)
                
        return assigned_centroid_dict
    
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Example usage
X = pd.DataFrame(np.random.rand(100, 2), columns=['x', 'y'])

kmeans = KMeans(k=3, max_iter=10)
kmeans.fit(X)

assigned_centroids = kmeans.predict(X)
print(assigned_centroids)

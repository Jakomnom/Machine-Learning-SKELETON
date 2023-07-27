import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, k, max_iter):
        self.k = k
        self.max_iter = max_iter
        
    def _compute_centroid(self, X):
        
        self.centroids = []
        for centroid, centroid_df in assigned_centroid.dict.items():
            centroid_mean = pd.DataFrame(centroid_df.mean(axis = 0))
            centroid_mean = centroid_mean.T
            
            self.centroids.append(centroid_mean)
            
        return pd.concat(self.centroids)
    
    def fit(self):
        
        self.centroids = X.sample(self.K)
        
        for i in range(self.max_iter):
            self.assigned_centroid_dict = self.predict(X)
            self.centroids = self.compute_centroid(X, self.assigned_centroid_dict)
            
    def predict(self):
        
        assigned_centroid_dict = {}
        
        for row_num, row_X in X.iterrows():
            
            row_X_df = pd.DataFrame(row_X).T
            
            assigned_centroid = None
            closest_distance = None
            
            for centroid_num, row_c in self.centroids.iterrows():
                
                distance = euclidian_distance(row_X, row_c)
                
                if assigned_centroid is None:
                    assigned_centroid = centroid_num
                    closest_distance = distance
                    continue
                elif distance < closest_distance:
                    assigned_centroid = centroid_num
                    closest_distance = distance
                    
            if assigned_centroid not in assigned_centroid_dict.keys():
                assigned_centroid_dict[assigned_centroid] = row_X_df
            else:
                assigned_centroid_dict[assigned_centroid].append(row_X_df)
                
        return assigned_centroid_dict    
        pass
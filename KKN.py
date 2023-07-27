import numpy as np
import pandas as pd

class KNN:
    def __init__(self, n_neighbors):
        self.n_neighbors
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def euclidean_distance(self, X1, X2):
        all_distances = {}
        for i_pred, X_pred in X.iterrows():
            individual_distances = {}
            for i_fit, X_fit in X.iterrows():
                distance = self.euclidean_distance(X_pred, X_fit)
                individual_distances[i_fit] = distance
                
            all_distances[i_pred] = individual_distances
        return all_distances
    
    def _select_neighbors(self, all_distances):
        nn_dict = {}
        
        for key, distances in all_distances.items():
            sorted_d = sorted(distances.items(), key = lambda item: item[1], reverse = True)
            nearest_neighbors = sorted_d[:self.n_neighbors]
            nn_dict[key] = nearest_neighbors
            
        return nn_dict
    
    def predict(self, X):
        
        all_distances = self._compute_distances(X)
        
        nn_dict = self._select_neighbors(all_distances)
        
        predictions = []
        for key, neighbors in nn_dict.items():
            labels = [self.y[neighbor[0]] for neighbor in neighbors]
            predictions.append(np.mean(labels))
        return predictions
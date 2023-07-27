import numpy as np
from scipy.spatial import distance

class KNN:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y):
        assert len(X) == len(y), "Input arrays X and y must have the same length."
        self.X = X
        self.y = y
        
    def euclidean_distance(self, X1, X2):
        return distance.cdist(X1, X2, 'euclidean')
    
    def predict(self, X):
        all_distances = self.euclidean_distance(self.X, X)
        predictions = []
        
        for i in range(X.shape[0]):
            distances = all_distances[:, i]
            sorted_indices = np.argsort(distances)
            nearest_indices = sorted_indices[:self.n_neighbors]
            nearest_labels = self.y[nearest_indices]
            prediction = np.mean(nearest_labels)
            predictions.append(prediction)
        
        return predictions

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

class GradientBoostedRegressor:
    def __init__(self, n_estimators, learning_rate, max_depth):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        self.avg_y = np.mean(y)
        
        for i in range(self.n_estimators):
            residual = y - self.predict(X)
            
            tree = DecisionTreeRegressor(self.max_depth)
            tree.fit(X, residual)
            
            residual_preds = tree.predict(X)
            
            self.trees.append(tree)
            
    def predict(self, X):
        final_prediction = np.full((1, len(X)), self.avg_y)[0]
        
        for tree in self.trees:
            resid_pred = tree.predict(X)
            final_prediction += resid_pred * self.learning_rate
            
        return final_prediction

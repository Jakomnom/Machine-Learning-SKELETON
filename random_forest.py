import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, max_samples, min_samples_split, max_depth, max_features, n_estimators = 100):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.min_samples_split = split
        self.max_depth = max_depth
        
        self.trees = []
        
    def fit(self, X, y):
        
        if self.max_creatures = 'auto':
            max_features = int(np.sqrt(len(X.columns)))
        
        #loop to create multiple decision trees
        for i in range(self.n_estimators):
            
            #fit a decision tree for each one of these estimators
            tree = DecisionTreeClassifier(max_depth = self.max_depth, min_samples_split = self.min_samples_split, max_features = self.max_features)
            
            #define a sub-sample
            indices = np.random.choice(X.shape[0], max_samples, replace = True)
            tree.fit(X.iloc[indices], y.iloc[indices])
            
            self.trees.append(tree)
            
    def predict(self, X):
        #basically pull the majority class to make our prediction
        all_preds = []
        for tree in self.trees:
            preds = tree.predict(X)
            all_preds.append(preds)
            
        return pd.DataFrame(all_preds).mean().values
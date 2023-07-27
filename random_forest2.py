import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, max_samples, min_samples_split, max_depth, max_features, n_estimators=100):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
        self.trees = []
        
    def fit(self, X, y):
        # Handle missing values (replace them with a specific value or strategy)
        X = X.fillna(0)  # Example: Replace missing values with zero
        
        if self.max_features == 'auto':
            max_features = int(np.sqrt(len(X.columns)))
        
        # Loop to create multiple decision trees
        for i in range(self.n_estimators):
            
            # Fit a decision tree for each estimator
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            
            # Define a sub-sample
            indices = np.random.choice(X.shape[0], self.max_samples, replace=True)
            tree.fit(X.iloc[indices], y.iloc[indices])
            
            self.trees.append(tree)
            
    def predict(self, X):
        # Handle missing values (replace them with a specific value or strategy)
        X = X.fillna(0)  # Example: Replace missing values with zero
        
        # Make predictions using the ensemble of decision trees
        all_preds = []
        for tree in self.trees:
            preds = tree.predict(X)
            all_preds.append(preds)
            
        # Return the mean predictions across all trees
        return pd.DataFrame(all_preds).mean().values
    
    def calculate_feature_importance(self, X, y):
        # Calculate feature importance based on the trained random forest classifier
        feature_importances = np.zeros(X.shape[1])
        for tree in self.trees:
            feature_importances += tree.feature_importances_
        feature_importances /= len(self.trees)
        return feature_importances
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, param_grid):
        # Perform hyperparameter tuning using grid search and cross-validation
        best_score = float("-inf")
        best_params = {}
        
        for params in param_grid:
            # Set the hyperparameters for the random forest classifier
            self.max_depth = params['max_depth']
            self.min_samples_split = params['min_samples_split']
            self.max_features = params['max_features']
            
            # Fit the random forest classifier on the training set
            self.fit(X_train, y_train)
            
            # Evaluate the model on the validation set
            score = self.evaluate(X_val, y_val)
            
            # Update the best parameters and score if a better result is obtained
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params, best_score
    
    def evaluate(self, X, y):
        # Evaluate the performance of the random forest classifier
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

class GradientBoostedRegressor:
    def __init__(self, n_estimators, learning_rate, max_depth):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X_train, y_train):
        self.avg_y = np.mean(y_train)
        
        for i in range(self.n_estimators):
            residual = y_train - self.predict(X_train)
            
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_train, residual)
            
            residual_preds = tree.predict(X_train)
            
            self.trees.append(tree)
            
    def predict(self, X):
        final_prediction = np.full((1, len(X)), self.avg_y)[0]
        
        for tree in self.trees:
            resid_pred = tree.predict(X)
            final_prediction += resid_pred * self.learning_rate
            
        return final_prediction

# Load and preprocess the data
data = pd.read_csv('demand_for_products.csv')  # Assuming data is in a CSV file

# Perform feature engineering and preprocessing steps

# Split the data into training and testing sets
X = data.drop('target', axis=1)  # Assuming 'target' column contains the data for demand
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the GradientBoostedRegressor
gb_regressor = GradientBoostedRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# Fit the model on the training data
gb_regressor.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = gb_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r_squared = 1 - (mse / np.var(y_test))

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r_squared}")

# Use the model to make predictions on new or unseen data
new_data = pd.read_csv('new_data.csv')  # Assuming new data is in a CSV file
new_predictions = gb_regressor.predict(new_data)

import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        """
        Naive Bayes classifier.
        """
        pass
    
    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier to the training data.
        
        Parameters:
        - X: Training samples, numpy array or pandas DataFrame of shape (n_samples, n_features).
        - y: Target values, numpy array or pandas Series of shape (n_samples,).
        """
        # Input validation
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        n_samples, n_features = X.shape
        
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.priors = np.zeros(n_classes)
        for i in range(n_classes):
            self.priors[i] = np.sum(y == self.classes[i]) / n_samples
            
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        for i in range(n_classes):
            X_class = X[y == self.classes[i]]
            self.means[i, :] = X_class.mean(axis=0)
            self.variances[i, :] = X_class.var(axis=0)
            
    def predict(self, X):
        """
        Predict the class labels for new samples.
        
        Parameters:
        - X: Test samples, numpy array or pandas DataFrame of shape (n_samples, n_features).
        
        Returns:
        - y_pred: Predicted class labels, numpy array of shape (n_samples,).
        """
        # Input validation
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        y_pred = []
        
        for sample in X:
            posteriors = []
            for i in range(len(self.classes)):
                prior = np.log(self.priors[i])
                posterior = np.log(self.calculate_likelihood(sample, self.means[i, :], self.variances[i, :])).sum()
                posterior = prior + posterior
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
            
        return np.array(y_pred)
            
    def calculate_likelihood(self, x, mean, var):
        """
        Calculate the likelihood of a feature value given the class.
        
        Parameters:
        - x: Feature value, scalar or numpy array of shape (n_features,).
        - mean: Mean values of the feature in the class, numpy array of shape (n_features,).
        - var: Variance values of the feature in the class, numpy array of shape (n_features,).
        
        Returns:
        - likelihood: Likelihood of the feature value given the class, scalar or numpy array of shape (n_features,).
        """
        exponent = np.exp(-((x - mean) ** 2 / (2 * var)))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

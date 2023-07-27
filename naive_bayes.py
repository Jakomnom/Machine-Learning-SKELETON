import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.priors = np.zeros(n_classes)
        for i in range(n_classes):
            self.priors[i] = np.sum(y==self.classes[i])/n_samples
            
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        for i in range(n_classes):
            X_class = X[y == self.classes[i]]
            self.means[i, :] = X_class.mean(axis = 0)
            self.variances[i, :] = X_class.var(axis = 0)
            
    def predict(self, X):
        
        y_pred = []
        
        for sample in X:
            posteriors = []
            for i in range(len(self.classes)):
                prior = np.log(self.priors[i])
                posterior = np.log(self.calculate_likelihood(sample, self.means[i, :], self.variances[i, :])).sum()
                posterior= prior + posterior
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
            
        return y_pred
            
    def calculate_likelihood(self, x, mean, var):
        
        exponent = np.exp(-((x - mean) ** 2 / (2 * var)))
        return (1/np.sqrt(x*np.pi*var)*exponent)
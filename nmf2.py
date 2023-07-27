import numpy as np
import pandas as pd

class NMF:
    def __init__(self, n_components, max_iter, err_lim):
        """
        Initialize the NMF object.
        
        Args:
            n_components (int): The number of components to extract.
            max_iter (int): The maximum number of iterations for the algorithm.
            err_lim (float): The error limit used as a convergence criterion.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.err_lim = err_lim
        
    def fit(self, X):
        """
        Perform NMF on the input matrix X.
        
        Args:
            X (numpy.ndarray): The input data matrix of shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape
        
        # Validate input shape
        if n_samples <= 0 or n_features <= 0:
            raise ValueError("Invalid input shape. Expected non-zero dimensions.")
        
        self.W = np.random.rand(n_samples, self.n_components)
        self.H = np.random.rand(self.n_components, n_features)
        
        for i in range(self.max_iter):
            self.H *= np.dot(self.W.T, X) / (np.dot(np.dot(self.W.T, self.W), self.H) + 1e-10)
            self.W *= np.dot(X, self.H.T) / (np.dot(np.dot(self.W, self.H), self.H.T) + 1e-10)
            
            reconstruction_err = np.mean((X - np.dot(self.W, self.H))**2)
            
            if i % 10 == 0:
                print("Iteration {}: error = {}".format(i, reconstruction_err))
                
            if reconstruction_err < self.err_lim:
                print("Convergence achieved at {} iterations".format(i))
                break
    
    def transform(self, X):
        """
        Apply the learned transformation to the input matrix X.
        
        Args:
            X (numpy.ndarray): The input data matrix of shape (n_samples, n_features).
        
        Returns:
            numpy.ndarray: The transformed matrix of shape (n_components, n_samples).
        """
        return np.dot(self.W.T, X)
    
    def fit_transform(self, X):
        """
        Fit the NMF model to the input matrix X and return the transformed matrix.
        
        Args:
            X (numpy.ndarray): The input data matrix of shape (n_samples, n_features).
        
        Returns:
            numpy.ndarray: The transformed matrix of shape (n_components, n_samples).
        """
        self.fit(X)
        return self.transform(X)

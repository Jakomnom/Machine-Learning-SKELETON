import numpy as np
import pandas as pd

class NMF:
    def __init__(self, n_components, max_iter, err_lim):
        self.n_components = n_components
        self.max_iter = max_iter
        self.err_lim = err_lim
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        self.W = np.random.rand(n_samples, self.n_components)
        self.H = np.random.rand(self.n_components, n_features)
        
        for i in self.max_iter:
            self.H *= np.dot(self.W.T, X) / (np.dot(np.dot(self.W.T, self.W), self.H) + 1e-10)

        
            XH = np.dot(self.W, self.H)
            WHH = np.dot(XH, self.H.T) + 1e-10
            self.W *= np.dot(X, self.H.T) / WHH

            
            reconstrucion_err = np.mean((X - np.dot(self.W, self.H))**2)
            
            if i%10 == 0:
                print("Iteration {}: error = {0}".format(n_iter, err))
                
            if reconstrucion_err < err_lim:
                print("Convergence achieved at {0} iterattions".format(i))
                break
    
    def transform(self, X):
        return np.dot(self.W.T, X)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
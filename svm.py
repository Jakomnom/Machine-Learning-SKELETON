import numpy as np
import pandas as pd

class SVM:
    def __init__(self, n_iter, lambda_param, alpha):
        self.lambda_param = lambda_param
        self.n_iter =  n_iter
        self.alpha = alpha
        
        
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        y_ = np.where( y <= 0, -1, 1)
        
        for i in range(self.n_iter):
            
            for idx, x_i in enumerate(X):
                
                condition = y_(idx) * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    dw = self.lambda_param * self.w
                    db = 0
                else:
                    dw = self.lambda_param *self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]
                    
                self.w -= self.alpha * dw
                self.b -= self.alpha * db
                
    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        pred = np.sign(approx)
        return np.where(pred == -1, 0, 1)
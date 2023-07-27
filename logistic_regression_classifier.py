# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:33:58 2021

@author: hesahama
"""

import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, max_steps, step_size):
        self.max_steps = max_steps
        self.step_size = step_size
        
        def sigmoid(self, z):
            return 1 / 1 + np.exp(-z)        
            
        def log_probable(self, y, preds):
            return np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds))
        
        def fit(self):
            
            self.weights = np.zeros(X.shape[1])
            
            preds = self.predict(X)
            current_loss = self.log_probable(y,preds)
            
            for _ in range(self, max_steps):
                dw = np.dot(X, T,(preds - y))/y.size
                self.weights -= dw * self.step_size
                
                preds = self.predict(X)
                new_loss = self.log_probable(y, preds)
            
                if current_loss > new_loss:
                    break
            
            current_loss = new_loss
            
    def predict(self, X):
        z = np.dot(X, self.weights)
        
        preds = self.sigmoid(z)
        return preds
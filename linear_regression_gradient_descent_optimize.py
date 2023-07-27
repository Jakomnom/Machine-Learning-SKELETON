# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 10:28:09 2022

@author: hesahama
"""

import numpy as np
import pandas as pd


    #atur konfigurasi awal untuk objek linearRegression
class linearRegression:
    def __init__(self, step_size, max_steps = 100):
        self.max_steps = max_steps
        self.step_size = step_size
        
    #jumlah perbedaan kuadrat
        
    def sum_of_squared_errors(self, y, preds):
        return np.sum((preds - y) ** 2)
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        
        self.weights = np.zeros(X.shape[1])
        
        preds = self.predict(X)
        
        current_loss = self.sum_of_squared_errors(X, y, preds)
        
        for _ in range (max_steps):
            dw = np.dot(X, T, (preds - y)) * (1/num_samples)
            
            self.weights -= dw * self.step_size
            
            preds = self.predict(self.weights)
            
            new_loss = self.sum_of_squared_errors(y, preds)
            
            if new_loss > current_loss:
                break
            
            current_loss = new_loss
            
    def predict(self, X):
        preds = np.dot(X, self.weights)
        return preds
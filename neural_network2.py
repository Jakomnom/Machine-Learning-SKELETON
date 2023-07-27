import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.hidden_weights = np.random.random((num_inputs, num_hidden))
        self.hidden_bias = np.zeros((1, num_hidden))
        self.output_weights = np.random.random((num_hidden, num_outputs))
        self.output_bias = np.zeros((1, num_outputs))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward_pass(self, inputs):
        hidden_layer = np.dot(inputs, self.hidden_weights) + self.hidden_bias
        self.hidden_layer_activation = self.sigmoid(hidden_layer)
    
        output_layer = np.dot(self.hidden_layer_activation, self.output_weights) + self.output_bias
        output_layer_activation = self.sigmoid(output_layer)
        
        return output_layer_activation
        
    def backward_pass(self, inputs, targets, outputs, learning_rate):
        output_error = targets - outputs
        output_delta = output_error * self.sigmoid_derivative(outputs)
        
        hidden_error = np.dot(output_delta, self.output_weights.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_activation)
        
        self.output_weights += learning_rate * np.dot(self.hidden_layer_activation.T, output_delta)
        self.hidden_weights += learning_rate * np.dot(inputs.T, hidden_delta)
        
        self.output_bias += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.hidden_bias += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
        
    def fit(self, inputs, targets, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            outputs = self.forward_pass(inputs)
            self.backward_pass(inputs, targets, outputs, learning_rate)
            
            mse = np.mean(np.square(targets - outputs))
            print("Epoch:", epoch, "MSE:", mse)
            
    def predict(self, inputs):
        return self.forward_pass(inputs)

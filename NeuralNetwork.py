import numpy as np
import datetime
import random

class NeuralNetwork():
    def __init__(self):
        np.random.seed(random.seed(datetime))
        
        self.weight_hidden = np.random.rand(25, 10)
        self.weight_output = np.random.rand(10, 1)
        self.learning_rate = 0.05
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivation(self, x):
        return x * (1 - x)
    
    def train(self, input_feature, target_output, iterations=1000):
        for epoch in range(iterations):
            input_hidden = np.dot(input_feature, self.weight_hidden)
            output_hidden = self.sigmoid(input_hidden)
            
            input_output = np.dot(output_hidden, self.weight_output)
            output = self.sigmoid(input_output)
            
            error_output = ((1/ 2) * np.power((output - target_output), 2))
            
            derror_output = output - target_output
            doutput_dino = self.sigmoid_derivation(input_output)
            dino_weight_output = output_hidden
            derror_weight_output = np.dot(dino_weight_output.T, derror_output * doutput_dino)
            
            derror_dino = derror_output * doutput_dino
            dino_douth = self.weight_output
            derror_douth = np.dot(derror_dino, dino_douth.T)
            douth_dinh = self.sigmoid_derivation(input_hidden)
            dinh_weight_hidden = input_feature
            derror_weight_hidden = np.dot(dinh_weight_hidden.T, douth_dinh * derror_douth)
            
            self.weight_hidden -= self.learning_rate * derror_weight_hidden
            self.weight_output -= self.learning_rate * derror_weight_output
            
    def predict(self, input_feature):
        result = np.dot(input_feature, self.weight_hidden)
        result = self.sigmoid(result)
        result = np.dot(result, self.weight_output)
        result = self.sigmoid(result)
        return result


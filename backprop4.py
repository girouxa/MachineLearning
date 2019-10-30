# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 20:56:46 2019

@author: Annie
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def randMat (d1,d2):
    return (2 * np.random.random_sample((d1,d2)) - 1)
def randVect(d1):
    return np.array(np.random.randint(5,size=(d1)))
def randVecMat(num, dims):
    a = []
    for i in range(num-1):
        print(a)
        a.append(randMat(dims[i+1], dims[i]))
    return a
def genBiases(num, nodes):
    a = []
    for i in range(num-1):
        a.append(2 * np.random.random_sample((nodes[i+1],1)) - 1)
    return a

class NeuralNetwork:
    def __init__(self, n):
        self.nodes = n
        self.weights = randVecMat(len(self.nodes), np.asarray(self.nodes.reverse()))
        self.biases = genBiases(len(n), self.nodes)
        self.zs = None
        self.activations = None
        pass
    def activation(self, x):
        if(len(x.shape) == 1):
            for k in range(x.shape[0]):
                x[k] = np.fmax(0,x[k])
            return x
        else:
            for j in range(x.shape[0]):
                for k in range(x.shape[1]):
                    x[j][k] = np.fmax(0, x[j][k])
            return x
    def activationDer(self,x):
        if(len(x.shape) == 1):
            for k in range(x.shape[0]):
              if(np.fmax(0,x[k]) < 0):
                x[k] = 0
              else:
                x[k] = 1
            return x      
        else:
            for j in range(x.shape[0]):
                for k in range(x.shape[1]):
                  if(np.fmax(0, x[j][k]) < 0):
                    x[j][k] = 0
                  else:
                    x[j][k] = 1
            return x
    def forward(self, x):
        pass
    def cost(self, x, y):
        return (np.sum(np.square(self.forward(x)-y))) #fix make 1/2n later
    def findGrad(self, y):
        pass
    
g = NeuralNetwork([2,1,3])
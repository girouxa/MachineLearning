# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:10:18 2019

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
        self.weights = randVecMat(len(self.nodes), self.nodes)
        self.biases = genBiases(len(n), self.nodes)
        self.zs = None
        self.activations = None
        self.newW = None
        self.newB = None
        pass
    def activation(self, x):
        """
        if(len(x.shape) == 1):
            for k in range(x.shape[0]):
                x[k] = 1/(1+np.exp(-1*x[k]))
            return x
        else:
            for j in range(x.shape[0]):
                for k in range(x.shape[1]):
                    x[j][k] = 1/(1+np.exp(-1*x[j][k]))
            return x
            """
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
        self.zs = []
        self.activations = []
        currentm = self.weights[0]
        sec = x
        for i in range(len(self.nodes)-1):
            weighted = currentm.dot(sec) + self.biases[i]
            self.zs.append(weighted)
            sec = self.activation(np.array(weighted, dtype = np.float64))
            self.activations.append(sec);
            if(i == (len(self.nodes)-2)):
                return sec
            else:
                currentm = self.weights[i+1]
    def cost(self, x, y):
        return (1/2*len(x))*(np.sum(np.square(self.forward(x)-y))) #fix make 1/2n later
    def backProp(self, y):
        layer = (len(self.weights) - 1)
        newW = []*(len(self.weights)-1)
        newB = []*(len(self.biases)-1)
        mat = (self.activations[layer]*self.activationDer(self.zs[layer])*(2*(self.activations[layer]-y)))
        bias = (self.activationDer(self.zs[layer])*(2*(self.activations[layer]-y)))
        newW.append(mat)
        newB.append(bias)
        for l in reversed(range(len(self.weights)-1)):
            mat = self.weights[l+1]*self.activationDer(self.activations[l+1])*newW[l]
            print(mat)
            if(l < 0):
                newW[l-1] = mat
            else:
                break
        self.newW = newW
        self.newB = newB
        return
    def gradientDescent(self, learningRate, num):
        for r in range(num):
            for u in range(len(self.newW)):
                self.weights[u] -= learningRate*self.newW[u]
                self.biases[u] -= learningRate*self.newB[u]
            
        
          
        

g = NeuralNetwork([3,4,6])
g.forward(np.array([[1,1,1]]).T)
g.backProp(np.array([[.5,.5,.5,.5,.5,.5]]).T)
g.gradientDescent(0.1, 1000)
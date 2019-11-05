# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:26:34 2019

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
    print("a:")
    print(a)
    return a
  
def value_at(shape, row, col, value):
  a = np.zeros(shape)
  a[row,col] = value
  return a

# return copy of original array with delta added to the row,col element
def delta_at(a, row, col, delta):
  return a + value_at(a.shape, row, col, delta)

def oneDToTwoD(x):
  xfin = []
  for i in range(len(x)):
    xfin.append([])
    xfin[i].append(x[0])
  return np.asarray(xfin)

class NeuralNetwork:
    def __init__(self, n):
        self.nodes = n
        self.weights = randVecMat(len(self.nodes), self.nodes)
        self.biases = genBiases(len(n), self.nodes)
        self.zs = None
        self.activations = None
        self.dW = None
        self.newB = None
        self.s  = None
        self.errorsubt = None
        self.errordivis = None
        pass
    def activation(self, x):
      return 1/(1+np.exp(-1*x))

#         if(len(x.shape) == 1):
#             for k in range(x.shape[0]):
#                 x[k] = np.fmax(0,x[k])
#             return x
#         else:
#             for j in range(x.shape[0]):
#                 for k in range(x.shape[1]):
#                     x[j][k] = np.fmax(0, x[j][k])
#             return x

    def activationDer(self,x):

#         if(len(x.shape) == 1):
#             for k in range(x.shape[0]):
#               if(np.fmax(0,x[k]) < 0):
#                 x[k] = 0
#               else:
#                 x[k] = 1
#             return x      
#         else:
#             for j in range(x.shape[0]):
#                 for k in range(x.shape[1]):
#                   if(np.fmax(0, x[j][k]) < 0):
#                     x[j][k] = 0
#                   else:
#                     x[j][k] = 1
#             return x

      r = self.activation(x)
      return r*(1-r)
    def estimateDWeights(self, x, y, epsilon):
        dw = [np.zeros_like(w) for w in self.weights]
        for layer in range(len(self.weights)):
            rows, cols = self.weights[layer].shape
            for r in range(rows):
                for c in range(cols):
                    tw = self.weights[layer] # make a backup of the weights matrix
                    self.weights[layer] = delta_at(tw, r, c, epsilon)
                    c1 = self.cost(x,y)
                    self.weights[layer] = delta_at(tw, r, c, -epsilon)
                    c2 = self.cost(x,y)
                    self.weights[layer] = tw # restore the backup
                    dc = (c1-c2)/(2*epsilon)
                    dw[layer][r,c] = dc
        return dw

    def estimateDBiases(self, x, y, epsilon):
        db = [np.zeros_like(b) for b in self.biases]
        for layer in range(len(self.biases)):
            rows, cols = self.biases[layer].shape
            for r in range(rows):
                for c in range(cols):
                    tb = self.biases[layer] # make a backup of the bias matrix
                    self.biases[layer] = delta_at(tb, r, c, epsilon)
                    c1 = self.cost(x,y)
                    self.biases[layer] = delta_at(tb, r, c, -epsilon)
                    c2 = self.cost(x,y)
                    self.biases[layer] = tb # restore the backup
                    dc = (c1-c2)/(2*epsilon)
                    db[layer][r,c] = dc
        return db
    def forward(self, x):
        self.zs = []
        self.activations = []
        currentm = self.weights[0]
        sec = x
        for i in range(len(self.nodes)-1):
            weighted = currentm.dot(sec) + self.biases[i]
            self.zs.append(weighted)
            sec = self.activation(np.array(weighted, dtype = np.float64))
            self.activations.append(np.asarray(sec));
            if(i == (len(self.nodes)-2)):
                return sec
            else:
                currentm = self.weights[i+1]
    def cost(self,x, y):
        return (1/2)*(np.sum(np.square(self.forward(x)-y))) 
    def backProp(self,x, y):
        layer = (len(self.weights) - 1)
        dW = [np.zeros_like(w) for w in self.weights]
        s = [np.zeros_like(w) for w in self.weights]
        sense = self.activationDer(self.zs[layer])*(self.activations[layer]-y)
        s[layer] = sense
        #mat = (self.activatzns[layer]*self.activationDer(self.zs[layer])*(2*(self.activations[layer]-y)))
        for l in reversed(range(len(self.weights)-1)):
            sense = self.activationDer(self.zs[l])*(self.weights[l+1].T@s[l+1])
            s[l] = sense
        for i in range(0, len(s)):
            if(i==0):
                dW[i] = x.T*s[i]
            else:
                dW[i] = self.activations[i-1].T*s[i]

        self.dW = dW
        self.dB = s
        self.s = s
        return
    def update(self, learningRate, x,  y):
        for u in range(len(self.s)):
            self.weights[u] -= learningRate*self.dW[u]
            self.biases[u] -= learningRate*self.dB[u]
    def gradientTest(self, e, c):
        errorsubt = []
        errordivis = []
        for i in range(len(e)):
            errorsubt.append(e[i]-c[i])
            errordivis.append(e[i]/c[i])
        self.errorsubt = errorsubt
        self.errordivis = errordivis
        return
    def gradientDescent(self, learningRate, x, y):
        self.forward(x.T)
        self.backProp(x.T,y.T)
        self.update(.03, x, y)
    def batchTrain(self,batchSize,xs,ys):
        x = np.split(xs[0], len(xs[0])/batchSize)
        y = np.split(ys[0], len(xs[0])/batchSize)
        x = oneDToTwoD(x)
        y = oneDToTwoD(y)
        for t in range(len(x)):
            self.gradientDescent(.03, x[t], y[t])
        
            

        

            
        
          


#g = NeuralNetwork([3,2,2])
#g.weights = [np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]]), np.array([[0.7,0.9],[0.8,0.1]])]
#g.biases = [np.array([[0.5,0.5]]).T, np.array([[0.5,0.5]]).T]
#x = np.array([[1,4,5]]).T
#y = np.array([[.1,.05]]).T
#g.forward(x)
#g.backProp(x,y)
#print("calculated weight derivatives:")
#print(g.dW)
#print("estimated weight derivatives:")
#print(g.estimateDWeights(x,y,.00001))
#g.gradientTest(g.dW,g.estimateDWeights(x,y,.00001))
#print("sensitivities:")
#print(g.s)
#print("activations:")
#print(g.activations)
#print("zs:")
#print(g.zs)
#print("division:")
#print(g.errordivis)
#print("subt:")
#print(g.errorsubt)
#print(" ")
#print("cost at first:")
#print(g.cost(x,y))
#print("cost after descent:")
#g.gradientDescent(.01, 10000,x, y)
#print(g.cost(x,y))
nn = NeuralNetwork([1,5,5,1])

xtrain = np.array(np.arange(0,10,1)).reshape(-1, 1)
xtest = np.array(np.arange(-5,15,0.1)).reshape(-1, 1)
ytrain = np.sin(xtrain)
#nn.batchTrain(1, xtrain, ytrain)
#print(nn.dW)
#print(nn.estimateDWeights(oneDToTwoD(xtrain), oneDToTwoD(ytrain), .00001))


history = nn.batchTrain(1, xtrain, ytrain)

plt.plot(history)
plt.show()

ytest = nn.predict(xtest.T)
plt.scatter(xtest.flatten(), ytest.flatten())
plt.scatter(xtrain.flatten(), ytrain.flatten())
plt.xlim(-5, 15)
plt.ylim(-1.2, 1.2)
plt.show()

#print(g.activations)
#print(g.activationDer(g.zs[1]))
#print(2*(g.activations[1]-y))
#print(g.estimateDBiases(x,y,.00001))
#print(g.newW)

#g.backProp(y)
#print(g.estimateDWeights(x, y, .000001))
#print(g.newW)
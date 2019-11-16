# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 12:22:43 2019

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
        self.vdw = [np.zeros_like(w) for w in self.weights]
        self.vdb = [np.zeros_like(b) for b in self.biases]
        pass
    def activation(self, x, relu):
        if(relu):
            return np.fmax(.01*x,x)
        else:
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

    def activationDer(self, x, relu):
        if(relu):
            return np.where(x <= 0, 0.01, 1)
        else:
            r = self.activation(x, False)
            return r*(1-r) 

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

      
    def estimateDWeights(self, x, y, epsilon):
        dw = [np.zeros_like(w) for w in self.weights]
        for layer in range(len(self.weights)):
            rows, cols = self.weights[layer].shape
            for r in range(rows):
                for c in range(cols):
                    tw = self.weights[layer] # make a backup of the weights matrix
                    self.weights[layer] = delta_at(tw, r, c, epsilon)
                    c1 = self.cost(x,y, .0001)
                    self.weights[layer] = delta_at(tw, r, c, -epsilon)
                    c2 = self.cost(x,y, .0001)
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
                    c1 = self.cost(x,y, .0001)
                    self.biases[layer] = delta_at(tb, r, c, -epsilon)
                    c2 = self.cost(x,y, .0001)
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
            if(i == (len(self.nodes)-2)):
              self.activations.append(np.asarray(weighted))
              return weighted
            else:
              sec = self.activation(np.array(weighted, dtype = np.float64), True)
              self.activations.append(np.asarray(sec));
            if(i == (len(self.nodes)-2)):
                return sec
            else:
                currentm = self.weights[i+1]
    def cost(self,x, y, lam):
        reg = 0
        for i in range(len(self.nodes)-1):
          reg += np.sum(np.square(self.weights[i]))
        return np.mean((1/2)*(np.sum(np.square(self.forward(x)-y), axis=0))) + (lam/2*y.shape[1])
    def backProp(self,x, y):
        self.forward(x)
        layer = (len(self.weights) - 1)
        dW = [np.zeros_like(w) for w in self.weights]
        s = [np.zeros_like(w) for w in self.weights]
        sense = self.activationDer(self.zs[layer], True)*(self.activations[layer]-y)
        s[layer] = sense
        #mat = (self.activatzns[layer]*self.activationDer(self.zs[layer])*(2*(self.activations[layer]-y)))
        for l in reversed(range(len(self.weights)-1)):
            sense = self.activationDer(self.zs[l], True)*(self.weights[l+1].T@s[l+1])
            s[l] = sense
        self.dB = []
        for i in range(0, len(s)):
            self.dB.append(np.mean(s[i], axis=1, keepdims=True))
            if(i==0):
                dW[i] = ((s[i]@x.T)/s[i].shape[1])
            else:
                dW[i] = ((s[i]@self.activations[i-1].T)/s[i].shape[1])

        self.dW = dW
        self.s = s
        return
    def update(self, learningRate, bet):
        for u in range(len(self.weights)):
            self.vdw[u] = bet*self.vdw[u] + (1-bet)*self.dW[u]
            self.vdb[u] = bet*self.vdb[u] + (1-bet)*self.dB[u]
            self.weights[u] = self.weights[u] - (learningRate*self.vdw[u])
            self.biases[u] = self.biases[u] - (learningRate*self.vdb[u])
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
        self.update(.03, 0.9)
    def fit(self, batchSize, epochs, learningRate, xs, ys):
        costs = []
        if batchSize == 0:
            batchSize = xs.shape[1]
        xs = np.copy(xs)
        for j in range(epochs):
            i = 0
            while i < xs.shape[1]:
                self.backProp(xs[:,i:i+batchSize], ys[:,i:i+batchSize])
                self.update(learningRate, 0.9)
                i += batchSize
            costs.append(self.cost(xs,ys, .0001))
        return costs
#        x = np.split(xs[0], len(xs[0])/batchSize)
#        y = np.split(ys[0], len(xs[0])/batchSize)
#        x = oneDToTwoD(x)
#        y = oneDToTwoD(y)
#        costs = []
#        for t in range(len(x)):
#            self.gradientDescent(.03, x[t], y[t])
#            costs.append(self.cost(x[t].T, y[t].T))
#        return costs
            
            
def test(sampleX, sampleY):
  skynet = NeuralNetwork([3,2,2])

  sampleW1 = np.array([
                      [0.1, 0.3, 0.5],
                      [0.2, 0.4, 0.6]
                      ])

  sampleW2 = np.array([
                      [0.7, 0.9],
                      [0.8, 0.1]
                      ])

  sampleB1 = np.array([[.5], [.5]])

  sampleB2 = np.array([[.5], [.5]])

  skynet.weights[0] = sampleW1
  skynet.weights[1] = sampleW2
  skynet.biases[0] = sampleB1
  skynet.biases[1] = sampleB2
  
  skynet.forward(sampleX.T)
  skynet.backProp(sampleX.T, sampleY.T)
  dwActual = skynet.dW
  print("actualdw:")
  print(dwActual)
  dbActual = skynet.dB
  print("actualdb:")
  print(dbActual)

  dwEstimate = skynet.estimateDWeights(sampleX.T, sampleY.T, 0.00001)
  dbEstimate = skynet.estimateDBiases(sampleX.T, sampleY.T, 0.00001)
  print("dwEstimate:")
  print(dwEstimate)
  print("dbEstimate:")
  print(dbEstimate)
  

  sumdiff = 0

  for wa,we in zip(dwActual, dwEstimate):
    d = wa-we
    sumdiff += np.sum(np.abs(wa-we))

  for ba,be in zip(dbActual, dbEstimate):
    d = ba-be
    sumdiff += np.sum(np.abs(ba-be))

  print(sumdiff)
  if sumdiff > 1e-9:
    print("Error in gradients!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  else:
    print("gradients look ok")



nn = NeuralNetwork([1,15,5,1])

xtrain = np.array(np.arange(0,10,1)).reshape(-1, 1)
xtest = np.array(np.arange(-5,15,0.1)).reshape(-1, 1)
ytrain = np.sin(xtrain)

history = nn.fit(1, 5000, 0.001, xtrain.T, ytrain.T)

plt.plot(history)
plt.show()

ytest = nn.forward(xtest.T)
plt.scatter(xtest.flatten(), ytest.flatten())
plt.scatter(xtrain.flatten(), ytrain.flatten())
plt.xlim(-5, 15)
plt.ylim(-1.2, 1.2)
plt.show()

#import math 
#
#layerlist =  [2,10,10,1]
#
#skynet = NeuralNetwork(layerlist)
#
#smile = np.array([[0,0,0,0,0,0,0,0,0,0],
#                  [0,0,0,0,0,0,0,0,0,0],
#                  [0,0,1,1,0,0,1,1,0,0],
#                  [0,0,1,1,0,0,1,1,0,0],
#                  [0,0,0,0,0,0,0,0,0,0],
#                  [0,0,0,0,0,0,0,0,0,0],
#                  [0,1,1,0,0,0,0,1,1,0],
#                  [0,0,1,1,0,0,1,1,0,0],
#                  [0,0,0,1,1,1,1,0,0,0],
#                  [0,0,0,0,0,0,0,0,0,0]
#                  ])
#
#xx, yy = np.mgrid[0:1:0.1, 0:1:0.1]
#x = np.c_[xx.ravel(), yy.ravel()]
#
#t = np.zeros((x.shape[0],1)) 
#
#for i in range(x.shape[0]):
#  t[i] = smile[int(x[i,0]*10),int(x[i,1]*10)]
#
#timage = t.reshape(len(xx),len(yy))
#
#cs = skynet.fit(1, 8000, 0.1, x.T, y.T)
#
#print(cs)
#print(math.sqrt(len(x)))
#f, ax = plt.subplots(figsize=(8, 6))
## ax.imshow(skynet.predict(x).reshape(len(xx),len(yy)))
#
#ax.plot(cs)
#
#f, ax = plt.subplots(figsize=(8, 6))
#ax.imshow(skynet.predict(x.T).reshape(len(xx),len(yy)))
#
#xx, yy = np.mgrid[0:1:0.01, 0:1:0.01]
#x = np.c_[xx.ravel(), yy.ravel()]
#
#f, ax = plt.subplots(figsize=(8, 6))
#ax.imshow(skynet.predict(x.T).reshape(len(xx),len(yy)))
#
#
#f, ax = plt.subplots(figsize=(8, 6))
#ax.imshow(timage)
#skynet = NeuralNetwork([3,2,2])
#
#sampleW1 = np.array([
#                    [0.1, 0.3, 0.5],
#                    [0.2, 0.4, 0.6]
#                    ])
#
#sampleW2 = np.array([
#                    [0.7, 0.9],
#                    [0.8, 0.1]
#                    ])
#
#sampleB1 = np.array([[.5], [.5]])
#
#sampleB2 = np.array([[.5], [.5]])
#
#skynet.weights[0] = sampleW1
#skynet.weights[1] = sampleW2
#skynet.biases[0] = sampleB1
#skynet.biases[1] = sampleB2
#
#
#
#sampleX = np.array([[1, 4, 5]])
#
#expectedY = np.array([[0.88955061], [0.80039961]])
#
#y = np.round(skynet.forward(sampleX.T), 8)
#
#print("You got: ")
#print(y)
#print("Should Be: ")
#print(expectedY)
#print("Error: ")
#print(np.sum(np.abs(expectedY - y)))
#
#sampleX = np.array([[1, 4, 5],
#                    [1, 7, 5]])
#
#expectedY = np.array([[0.88955061, 0.89039757], [0.80039961, 0.8014626 ]])
#
#y = np.round(skynet.forward(sampleX.T), 8)
#
#print("You got: ")
#print(y)
#print("Should Be: ")
#print(expectedY)
#print("Error: ")
#print(np.sum(np.abs(expectedY - y)))
#
#skynet.backProp(sampleX.T, expectedY.T)
    
#sampleX = np.array([[1, 4, 5]])
#sampleY = np.array([[0.1, 0.05]])
#
#test(sampleX, sampleY)
#
#
#sampleX = np.array([[1, 4, 5],
#                    [1, 7, 5]])
#sampleY = np.array([[0.1, 0.05], 
#                    [0.1, 0.05]])
#
#test(sampleX, sampleY)
#k = NeuralNetwork([3,2,2])
#p = k.fit(0, 3, .03, sampleX.T, sampleY.T)



        
            

        

            
        
          


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
#nn = NeuralNetwork([1,5,5,1])
#
#xtrain = np.array(np.arange(0,10,1)).reshape(-1, 1)
#xtest = np.array(np.arange(-5,15,0.1)).reshape(-1, 1)
#ytrain = np.sin(xtrain)
#nn.batchTrain(1, xtrain, ytrain)
#print(nn.dW)
#print(nn.estimateDWeights(oneDToTwoD(xtrain), oneDToTwoD(ytrain), .00001))
#
#
#history = nn.batchTrain(1, xtrain, ytrain)
#
#plt.plot(history)
#plt.show()
#
#ytest = nn.predict(xtest.T)
#plt.scatter(xtest.flatten(), ytest.flatten())
#plt.scatter(xtrain.flatten(), ytrain.flatten())
#plt.xlim(-5, 15)
#plt.ylim(-1.2, 1.2)
#plt.show()

#print(g.activations)
#print(g.activationDer(g.zs[1]))
#print(2*(g.activations[1]-y))
#print(g.estimateDBiases(x,y,.00001))
#print(g.newW)

#g.backProp(y)
#print(g.estimateDWeights(x, y, .000001))
#print(g.newW)
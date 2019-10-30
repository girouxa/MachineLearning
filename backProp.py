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
        a.append(randMat(dims[i], dims[i+1]))
    return a
def genBiases(num, nodes):
    a = []
    for i in range(num-1):
        a.append(np.random.random_sample((1,nodes[i+1])))
    return a

class NeuralNetwork:
    def __init__(self, n):
        self.nodes = n
        #self.weights = np.array(np.array(np.random.rand(n[0],n[1])),np.array(np.random.rand(n[1],n[2])))
        self.weights = randVecMat(len(self.nodes), self.nodes)
        self.biases = genBiases(len(n), self.nodes)
        self.zs = None
        self.activations = None
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
                x[k] = np.max(0,x[k])
            return x
        else:
            for j in range(x.shape[0]):
                for k in range(x.shape[1]):
                    x[j][k] = np.max(0, x[j][k])
            return x
    
    def forward(self, x):
        self.zs = []
        self.activations = []
        currentm = self.weights[0]
        sec = x
        for i in range(len(self.nodes)-1):
            weighted = sec.dot(currentm) + self.biases[i]
            self.zs.append(weighted)
            sec = self.activation(np.array(weighted, dtype = np.float64))
            self.activations.append(sec);
            if(i == (len(self.nodes)-2)):
                return sec
            else:
                currentm = self.weights[i+1]
    
"""
N = NeuralNetwork(np.array([4,8,4,5]))
x = np.array([[1,1,1,1], [2,3,4,1]])
y = N.forward(x)
print(y)

"""
layerlist = np.array([2,5,5,5,1])

skynet = NeuralNetwork(layerlist)

xx, yy = np.mgrid[-2:2:.02, -2:2:.02]
grid = np.c_[xx.ravel(), yy.ravel()]

y = [skynet.forward(g.reshape(1,-1)) for g in grid]
y = np.array(y)

y = y.reshape(200,200)

f, ax = plt.subplots(figsize=(8, 6))

ax.imshow(y)

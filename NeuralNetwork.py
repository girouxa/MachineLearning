import numpy as np
import pandas as pd
import seaborn as sns


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
    def __init__(self, n, x):
        self.nodes = n
        #self.weights = np.array(np.array(np.random.rand(n[0],n[1])),np.array(np.random.rand(n[1],n[2])))
        self.weights = randVecMat(len(self.nodes), self.nodes)
        self.biases = genBiases(len(n), self.nodes)
        self.output = None
        pass
    def activation(self, x):
        if(len(x.shape) == 1):
            for k in range(x.shape[0]):
                x[k] = 1/(1+np.exp(-1*x[k]))
            return x
        else:
            for j in range(x.shape[0]):
                print(x.shape[1])
                for k in range(x.shape[1]):
                    x[j][k] = 1/(1+np.exp(-1*x[j][k]))
            return x
    def mult(self,x):
        x.dot(self.weights[1]) 
    def forward(self,x):
        currentm = self.weights[0]
        sec = x
        for i in range(len(self.nodes)-1):
            weighted = sec.dot(currentm) + self.biases[i]
            sec = self.activation(np.array(weighted, dtype = np.float64))
            if(i == (len(self.nodes)-2)):
                self.output = sec
                return sec
            else:
                currentm = self.weights[i+1]
    

N = NeuralNetwork(np.array([3,2,4]), np.array([1,1,2]))
j = N.forward(np.array([1,1,2]))
print(N.output)

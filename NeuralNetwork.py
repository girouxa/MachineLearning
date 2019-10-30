import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt


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

def value_at(shape, row, col, value):
  a = np.zeros(shape)
  a[row,col] = value
  return a

# return copy of original array with delta added to the row,col element
def delta_at(a, row, col, delta):
  return a + value_at(a.shape, row, col, delta)

class NeuralNetwork:
    def __init__(self, n, x):
        self.input = x
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
    def forward(self):
        currentm = self.weights[0]
        sec = self.input
        for i in range(len(self.nodes)-1):
            weighted = sec.dot(currentm) + self.biases[i]
            sec = self.activation(np.array(weighted, dtype = np.float64))
            if(i == (len(self.nodes)-2)):
                self.output = sec
                return sec
            else:
                currentm = self.weights[i+1]
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
        db = [np.zeros_like(b) for b in self.b]
        for layer in range(len(self.b)):
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
    def cost(self, x, y):
        return (1/2*len(x))*(np.sum(np.square(self.forward(x)-y))) 
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
            if(l < 0):
                newW[l-1] = mat
            else:
                break
        self.newW = newW
        self.newB = newB
        return
    def gradientDescent(self, learningRate, num, y):
        for r in range(num):
            for u in range(len(self.newW)):
                self.weights[u] -= learningRate*self.newW[u]
                self.biases[u] -= learningRate*self.newB[u]
                self.backProp(y)
    


N = NeuralNetwork(np.array([4,8,4,5]), np.array([1,7,2,7]))
N.forward()
print(N.estimateDWeights)


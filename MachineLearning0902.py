# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 09:32:27 2019

@author: Annie
"""
import numpy as np

#y = (3,4,2)  tuple -> immutable (can't be modified)
#y = 1, 2, 3 also makes tuple
#y = 1, 1-tuple

#d = { 'x' : 1, 'y' : 7}


# np.hstack(tup) -> sticks arrays of same dim to others
#np.ones((matrix.shape[0], 1)) -> puts a 1 in front of all rows
def Ment(m):
     return np.hstack((np.ones((m.shape[0],1)),m))



 


class LinearRegression:
    def __init__(self):
        self.w = np.array([0,1])
        self.learningrate = 0.1
        pass
    def h(self,x):
        if self.w is None:
            raise Exception('Train')
            #peppa pig
        return np.dot(x,self.w)
    def cost(self,x,y):
        m=len(y)
        z=0
        for i in range (m):
            z += (self.h(x[i])-y[i])**2
        z=z/(2*m)
        return z
    def dCost(self,x,y):
        m = len(x)
        dz = [0]*len(x[0])
        for i in range(m):
            for j in range(len(x[0])):
                dz[j] +=(self.h(x[i])-y[i])*x[i][j]
        dz = [i/m for i in dz]
        return dz
    def gradientStep(self,x,y):
        d = self.dCost(x,y)
        return np.array([self.w[j]-self.learningrate*d[j] for j in range(len(self.w))])
    def train(self,x,y):
        for i in range(1000):
            self.w = self.gradientStep(x,y)
        return

x = np.array([[1,1],[1,2],[1,3],[1,4],[1,5]])
y = np.array([2,5,6,8,11])

lr = LinearRegression()
lr.train(x,y)
f = lr.cost(x,y)
z = lr.w   

            
        
    
        
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 09:32:27 2019

@author: Annie
"""
import numpy as np
import pandas as pd
import seaborn as sns
import math as math
import matplotlib.pyplot as plt
sns.set(style="white")

#np.choose(iris[:,4] == 'setosa', [0,1]) <- get 0s/1s
# y = (iris[:,4] == 'setosa').astype(int) <- nvm use this one
# x = iris[:,0:4]

#y = (3,4,2)  tuple -> immutable (can't be modified)
#y = 1, 2, 3 also makes tuple
#y = 1, 1-tuple

#d = { 'x' : 1, 'y' : 7}


# np.hstack(tup) -> sticks arrays of same dim to others
#np.ones((matrix.shape[0], 1)) -> puts a 1 in front of all rows
def Ment(m):
     return np.hstack((np.ones((m.shape[0],1)),m))



 
#typeinterrorfix

class LinearRegression:
    def __init__(self):
        self.w = None
        self.learningrate = 0.1
        pass
    def h(self,x):
        if self.w is None:
            raise Exception('Train')
            #peppa pig
        return 1/(1+np.exp(-1*np.dot(x,self.w)))
    def cost(self,x,y):
        m=len(y)
        z=0
        #for i in range (m):
        #     z += (self.h(x[i])-y[i])**2
        #z=z/(2*m)
        for i in range(m):
            z += -y[i]*(np.log(self.h(x[i]))) - (1-y[i])*(np.log(1-self.h(x[i])))
        z=z/m
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
        self.w=[1]*len(x[0])
        for i in range(1000):
            self.w = self.gradientStep(x,y)
        return

iris = sns.load_dataset('iris').values
y = (iris[:,4] == 'setosa').astype(int)
#x = iris[:,0:4].astype(np.float64)
x = iris[:,0:4].astype(np.float64)
Ment(x)
lr = LinearRegression()
lr.train(x,y)
f = lr.cost(x,y)
z = lr.w
f, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x[:,0],x[:,1], c=y, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)
ax.set(aspect="equal",
       xlim=(4, 8), ylim=(1.8, 4.5),
       xlabel="$X_1$", ylabel="$X_2$")

#xx, yy = np.mgrid[4:8:.1, 2:4.5:.1]
#grid = np.c_[xx.ravel(), yy.ravel()]
#probs = lr.predict_proba(grid)[:, 1].reshape(xx.shape)



#f, ax = plt.subplots(figsize=(8, 6))
#contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
#                     vmin=0, vmax=1)
#ax_c = f.colorbar(contour)
#ax_c.set_label("$P(y = 1)$")

#ax.scatter(x[:,0],x[:,1], c=y, s=50,
#           cmap="RdBu", vmin=-.2, vmax=1.2,
#           edgecolor="white", linewidth=1)
#ax.set(aspect="equal",
#       xlim=(4, 8), ylim=(1.8, 4.5),
#       xlabel="$X_1$", ylabel="$X_2$") 

#answers = lr.h(grid)

        
    
        
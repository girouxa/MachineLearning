import matplotlib as plt
 

w= [0,1]
x = [[1,5],[1,4],[1,3],[1,2],[1,1]]
y = [2,5,6,8,10]

def hypothesis(w,x):
    return w[0]*x[0] + w[1]*x[1]

def getCost(w,x,y):
    m=len(y)
    z=0
    for i in range (m):
        z += (hypothesis(w,x[i])-y[i])**2
    z=z/(2*m)
    return z

def dCost(w,x,y):
    m = len(x)
    dz = [0]*len(w)
    for i in range(m):
        for j in range(len(w)):
            dz[j] +=(hypothesis(w,x[i])-y[i])*x[i][j]
    dz = [i/m for i in dz] #list comprehension
    return dz

def gradientStep1(learningRate,w,x,y):
    d = dCost(w,x,y)
    return [w[j]-learningRate*d[j] for j in range(len(w))]
    
def gradientStep2(learningRate,w,x,y):
    d = dCost(w,x,y)
    newW = []
    for i in range(len(w)):
        newW.append(w[i]-learningRate*d[i])
    return newW

def descent(learningRate, w, x, y):
    for i in range(100):
        w = gradientStep1(learningRate,w,x,y)
    return w

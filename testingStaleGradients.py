import numpy as np
from collections import deque
import matplotlib.pyplot as plt
 
def testDelays(function, maxDelay=10):
    iterList = []
    for delay in range(1, maxDelay):
        gradient, vectorList, noIters = gd(gradient=function, start=10, learning_rate=0.2, delay=delay, n_iter=10000)
        print(noIters)
        iterList.append(noIters)
 
    plt.plot(range(1, maxDelay), iterList)
    plt.yscale('log')
    plt.ylabel('Number of iterations')
    plt.xlabel('Delay')
    plt.grid()
    plt.show()
 
def gd(gradient, start, learning_rate, n_iter=500, tol=1e-06, delay=2):
    '''
    Inputs:
    gradient: Function that takes a vector and returns a gradient of the function
    start: The initial point on the function
    learning rate: Controls the magnitude of each step
    n_iter: number of iterations
    '''
    allVectors = [start]
    q = deque()
    for _ in range(delay):
        q.append(start)
    newVector = start    
    nIters = n_iter
    for i in range(n_iter):
       
        vector = q.popleft()
        #Calculates step size
        diff = -learning_rate * gradient(vector)
       
        #Checks gradient against tolerance
        if np.all(np.abs(diff) <= tol):
            nIters = i
            break
        #Adds step to gradient
        newVector += diff
        q.append(newVector)
        allVectors.append(newVector)
       
    return vector, allVectors, nIters
 
function = (lambda v:2*v)
 
testDelays(function=function)
 
'''gradient, vectorList, noIters = gd(gradient=function, start=10, learning_rate=0.2, delay=2)
print(gradient)
print(noIters)'''
#print(vectorList)

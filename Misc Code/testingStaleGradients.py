import numpy as np
from sympy import Symbol, Derivative
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from matplotlib.ticker import LinearLocator

from tqdm import tqdm

 
def testDelays(function, maxDelay=5, learningRateInterval=10, maxIter=500):
    iterList = []

    y = np.arange(1, maxDelay)
    x = np.linspace(0.001, 0.5, num=learningRateInterval)
    X, Y = np.meshgrid(x,y)

    for rate in tqdm(x):
        for delay in y:
            noIters = gd(function=function, start=10, learning_rate=rate, delay=delay, n_iter=maxIter)
            iterList.append(noIters)

    Z = np.array(iterList).reshape(y.size, x.size)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='plasma')
    ax.set_xlabel('Learning Rate')
    #ax.set_zscale('log')
    ax.set_ylabel('Delay')
    ax.set_zlabel('Number of Iterations to Converge')
    plt.show()
 
def gd(function, start, learning_rate, n_iter=500, tol=1e-06, delay=1) -> int:
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
    nIters = 0

    gradient = Derivative(function, x).doit()

    for i in range(n_iter):
       
        vector = q.popleft()
        #Calculates step size
        
        diff = -learning_rate * gradient.subs({x:vector})
       
        #Checks gradient against tolerance
        if np.all(np.abs(diff) <= tol):
            nIters = i
            break
        #Adds step to gradient
        newVector += diff
        q.append(newVector)
        allVectors.append(newVector)

    return nIters

x = Symbol('x')
function = x**2 + x*4 + 5

maxDelay = 10
learningRateInterval=100
maxIter = 200

testDelays(function=function, maxDelay=maxDelay, learningRateInterval=learningRateInterval, maxIter=maxIter)


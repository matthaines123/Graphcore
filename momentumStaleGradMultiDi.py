from sympy import Symbol, Derivative
import numpy as np
import time
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from collections import deque
from sympy import sin, cos, exp, sqrt
from numpy import pi, e
from decimal import Decimal, DecimalException

class Plot3D():
    def __init__(self, noOfPoints, function, margin):
        self.numberOfPoints = noOfPoints
        self.f = function
        margin = margin
        self.x_min, self.y_min = 0.0 - margin, 0.0 - margin
        self.x_max, self.y_max = 0.0 + margin, 0.0 + margin
        self.x_points = np.linspace(self.x_min, self.x_max, self.numberOfPoints)
        self.y_points = np.linspace(self.y_min, self.y_max, self.numberOfPoints)
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_points, self.y_points)
        self.z = self.getMeshSolution()

    def getMeshSolution(self):
        count = 0
        funcValue = []
        for xi, yi in zip(self.x_mesh, self.y_mesh):
            for val in xi:
                funcValue.append(self.f.subs([(x, val),(y,yi[0])]))
        z = np.array(funcValue).reshape(self.numberOfPoints, self.numberOfPoints)
        return z
        #np.array([f.subs(xps, yps) for xps, yps in zip(x_mesh, y_mesh)])

    def plotMinima(self):
        
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d', elev=80, azim=-100)

        ax.plot_surface(self.x_mesh.astype(float), self.y_mesh.astype(float), self.z.astype(float), norm=LogNorm(), rstride=1, cstride=1, 
                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.set_xlim((self.x_min, self.x_max))
        ax.set_ylim((self.y_min, self.y_max))

        plt.show()

    def plotMinimaWithPath(self, path, opt):
        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection='3d', elev=80, azim=-100)

        ax.plot_surface(self.x_mesh.astype(float), self.y_mesh.astype(float), self.z.astype(float), norm=LogNorm(), rstride=1, cstride=1, 
                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
        funcStart = []
        funcEnd = []
        for i in range(0, len(path[0,:-1])):
            funcStart.append(opt.func([path[0, i], path[1, i]]))
        for i in range(0, len(path[0,:-1])):
            funcEnd.append(opt.func([path[0, i+1], path[1, i+1]])-opt.func([path[0, i], path[1, i]]))

        ax.quiver(path[0,:-1], path[1,:-1], funcStart,
          path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1],
          funcEnd,
          color='k', length=1, normalize=True)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.set_xlim((self.x_min, self.x_max))
        ax.set_ylim((self.y_min, self.y_max))

        plt.show()

    def contourPlotWithPath(self, path):
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.contour(self.x_mesh, self.y_mesh, self.z, levels=np.logspace(-.5, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
        
        ax.quiver(path[0,:-1].astype(float), path[1,:-1].astype(float), path[0,1:].astype(float)-path[0,:-1].astype(float), path[1,1:].astype(float)-path[1,:-1].astype(float),
                scale_units='xy', angles='xy', scale=1, color='k')

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_xlim((self.x_min, self.x_max))
        ax.set_ylim((self.y_min, self.y_max))

        plt.show()

    def contourPlot(self):
        fig, ax = plt.subplots(figsize=(20, 16))

        ax.contour(self.x_mesh.astype(float), self.y_mesh.astype(float), self.z.astype(float), levels=np.logspace(-.5, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_xlim((self.x_min, self.x_max))
        ax.set_ylim((self.y_min, self.y_max))

        plt.show()

    def plotContourWithMinima(self, path, zList, maxIter, velocities):        
        fig = plt.figure(figsize=plt.figaspect(0.8))
        ax = fig.add_subplot(2,2,1,projection='3d')

        ax.plot_surface(self.x_mesh.astype(float), self.y_mesh.astype(float), self.z.astype(float), norm=LogNorm(), rstride=1, cstride=1, 
                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.set_xlim((self.x_min, self.x_max))
        ax.set_ylim((self.y_min, self.y_max))

        ax = fig.add_subplot(2,2,2)

        ax.contour(self.x_mesh, self.y_mesh, self.z, levels=np.logspace(-.5, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
    
        ax.quiver(path[0,:-1].astype(float), path[1,:-1].astype(float), path[0,1:].astype(float)-path[0,:-1].astype(float), path[1,1:].astype(float)-path[1,:-1].astype(float), scale_units='xy', angles='xy', scale=1, color='k')

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_xlim((self.x_min, self.x_max))
        ax.set_ylim((self.y_min, self.y_max))

        ax = fig.add_subplot(2,2,3)

        x = np.linspace(0, maxIter, maxIter + 1)
        ax.set_yscale("log")
        ax.plot(x, zList)
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel('z')

        ax = fig.add_subplot(2,2,4)

        ax.plot(x, velocities[0], label="dx")
        ax.plot(x, velocities[1], label="dy")
        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel("Velocity in direction")
        plt.legend()

        plt.show()


class Optimiser():
    def __init__(self, function, varSymbols, varInits=None, tol=1e-7, learning_rate=0.01, momentum=0.9, variableMomentumScalar=None, delay=1):
        self.function = function
        self.varSymbols = varSymbols
        self.gradients = self.gradient(self.function)

        self.funcValue = None
        self.varValues = np.zeros([len(varSymbols)])
        self.diffValues = np.zeros([len(varSymbols)])

        scale = 3.0
        for varIndex in range(len(varSymbols)):
            if varInits is not None:
                self.varValues[varIndex] = varInits[varIndex]
            else:
                self.varValues[varIndex] = np.random.uniform(low=-scale, high=scale)

        self.tol = tol
        self.lr = learning_rate
        self.momentumVec = np.array([momentum] * len(self.varSymbols))
        if variableMomentumScalar != None:
            self.variableMomentumScalar = variableMomentumScalar
        else:
            self.variableMomentumScalar = None
        self.velocity = np.zeros([len(varSymbols)])

        if delay != None:
            self.q = deque()
            for _ in range(delay):
                self.q.append([self.velocity])

        self.varValuesHistory = [[] for var in range(len(varSymbols))]
        self.diffValuesHistory = [[] for var in range(len(varSymbols))]
        self.funcValHistory = []

    def gradient(self, function):
        return [Derivative(function, symbol).doit() for symbol in self.varSymbols]
    
    def update_weights(self, grads, velocity):
        velocity = np.multiply(self.momentumVec, velocity) + np.multiply(self.lr, grads)        
        self.varValues = np.subtract(self.varValues, velocity)[0]
        return velocity

    def historyUpdate(self, funcValue, varValues, diffValues):
        self.funcValHistory.append(funcValue)
        for varIndex, value in enumerate(varValues):
            self.varValuesHistory[varIndex].append(value)
        for varIndex, value in enumerate(diffValues):
            self.diffValuesHistory[varIndex].append(value)

    def func(self):
        varValuePairs = []
        for symbol, varValue in zip(self.varSymbols, self.varValues):
            varValuePairs.append((symbol, varValue))
        result = self.function.subs(varValuePairs)
        return result

    def grads(self):
        varValuePairs = []
        for symbol, varValue in zip(self.varSymbols, self.varValues):
            varValuePairs.append((symbol, varValue))
        allGradients = []
        for gradient in self.gradients:
            dn = gradient.subs(varValuePairs)
            allGradients.append(dn)
        return allGradients

    def getCurrentVelocity(self):
        return self.q.popleft()

    def saveVelocity(self, newVelocity):
        self.q.append(newVelocity)

    def createPath(self):
        x_history = np.array(self.varValuesHistory[0])
        y_history = np.array(self.varValuesHistory[1])
        velocities = [np.array(self.diffValuesHistory[0])[:-1], np.array(self.diffValuesHistory[1])[:-1]]
        path = np.concatenate((np.expand_dims(x_history, 1), np.expand_dims(y_history, 1)), axis=1).T
        return velocities, path

    def variableMomentum(self, step):
        for index, dnHistory in enumerate(self.diffValuesHistory):
            dn1 = self.diffValuesHistory[index][step]
            dn2 = self.diffValuesHistory[index][step - 1]
            if dn1 > 0 and dn2 < 0:
                self.momentumVec *= self.variableMomentumScalar
                return
            elif dn1 < 0 and dn2 > 0:
                self.momentumVec *= self.variableMomentumScalar
                return

    def train(self, maxIter):
        funcValueList = np.zeros(maxIter + 1)
        for step in range(maxIter):
            currentVelocity = self.getCurrentVelocity()
            funcValue = self.func()
            funcValueList[step+1] = funcValue

            diff = np.abs(funcValueList[step] - funcValueList[step+1])
            
            self.grad = self.grads()
            self.historyUpdate(funcValue, self.varValues, self.grad)
            newVelo = self.update_weights(self.grad, currentVelocity)

            if self.variableMomentumScalar is not None:
                self.variableMomentum(step)
            
            self.saveVelocity(newVelo)

            if np.abs(diff) < self.tol and step > 5:
                print("Enough convergence!")
                print("Steps: {} Function Value: {}, Diff:{}".format(step+1, self.func(), diff))
                self.historyUpdate(funcValue, self.varValues, self.grad)
                break

            if (step+1) % 100 == 0:
                try:
                    print("Steps: {} Function Value: {}, Diff:{}".format(step+1, self.func(), diff))
                except Exception:
                    pass

        velocities, path = self.createPath()
        return funcValueList, step, velocities, path

class OptimiserWithVaryingDelay(Optimiser):
    def __init__(self, function, varSymbols, varInits=None, tol=1e-7, learning_rate=0.01, momentum=0.9, variableMomentumScalar=None, meanDelay=1):
        super().__init__(function, varSymbols, varInits, tol, learning_rate, momentum, variableMomentumScalar, delay=None)
        self.meanDelay = meanDelay
        self.velocities = []
        self.velocities.append(self.velocity)

    def getProbability(self, meanDelay):
        mu = meanDelay
        sigma = 1
        dist = round(np.random.normal(mu, sigma))
        if dist <= 1:
            return 1
        else:
            return dist
        
    def getCurrentVelocity(self):
        delay = self.getProbability(self.meanDelay)
        if len(self.velocities) >= (delay + 1):
            return self.velocities[-delay]
        else:
            return self.velocities[0]

    def saveVelocity(self, newVelocity):
        self.velocities.append(newVelocity)

    def update_weights(self, grads, velocity):
        velocity = np.multiply(self.momentumVec, velocity) + np.multiply(self.lr, grads)
        self.varValues = np.subtract(self.varValues, velocity)
        return velocity

    def createPath(self):
        x_history = np.array(self.varValuesHistory[0])
        y_history = np.array(self.varValuesHistory[1])
        velocities = [np.array(self.diffValuesHistory[0])[:-1], np.array(self.diffValuesHistory[1])[:-1]]
        path = np.concatenate((np.expand_dims(x_history, 1), np.expand_dims(y_history, 1)), axis=1).T
        return velocities, path

x = Symbol('x')
y = Symbol('y')
varSymbols = [x, y]

""" Functions """
beale = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2   #1, 1.4
g = x**2 + y**2 + 4*x + 5*x
h = -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
goldsteinPrice = (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)) 
booth = (x+2*y-7)**2 + (2*x + y - 5)*2
sphere = x**2 + y**2
rosenbrock = (x-1)**2 + 10*(y-x**2)**2 #-1.4, -1
matyas = 0.26*(x**2+y**2)-0.48*x*y  #3, 3.5
levi = sin(3*pi*x)**2 + ((x-1)**2)*(1+sin(3*pi*y)**2) + ((y-1)**2)*(1+sin(2*pi*y))  #-3, 3
himmelblau = (x**2 + y - 11)**2 + (x+y**2-7)**2 #1, -3
""""""

varInits = [1, 1.2]
opt = OptimiserWithVaryingDelay(beale, varSymbols, varInits, tol=1e-7, learning_rate=0.02, momentum=0.8, variableMomentumScalar=0.95, meanDelay=30)
zList, convergeIter, velocities, path = opt.train(1000)

Plot3D = Plot3D(50, beale, margin=4.5)
#Plot3D.plotMinima()
#Plot3D.contourPlotWithPath(opt.path)
Plot3D.plotContourWithMinima(path, zList[:convergeIter+1], convergeIter, velocities)

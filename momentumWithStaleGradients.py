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

        x = np.linspace(0, maxIter, maxIter+1)
        ax.set_yscale("log")
        ax.plot(x, zList)
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel('z')

        ax = fig.add_subplot(2,2,4)

        try:
            ax.plot(x, velocities[0], label="dx")
            ax.plot(x, velocities[1], label="dy")
        except ValueError:
            ax.plot(x[:-1], velocities[0], label="dx")
            ax.plot(x[:-1], velocities[1], label="dy")
        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel("Velocity in direction")
        plt.legend()

        plt.show()


class Optimiser():
    def __init__(self, function, x_init=None, y_init=None, learning_rate=0.01, momentum=0.9, delay=1):
        self.function = function
        self.gradients = self.gradient(self.function)
        scale = 3.0
        self.vars = np.zeros([2])
        if x_init is not None:
            self.vars[0] = x_init
        else:
            self.vars[0] = np.random.uniform(low=-scale, high=scale)
        if y_init is not None:
            self.vars[1] = y_init
        else:
            self.vars[1] = np.random.uniform(low=-scale, high=scale)

        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = np.zeros([2])

        if delay != None:
            self.q = deque()
            for _ in range(delay):
                self.q.append([self.velocity])

        self.z_history = []
        self.x_history = []
        self.y_history = []
        self.dx_history = []
        self.dy_history = []
        self.oscillating = [False, False] #One for each axis [dx, dy]

    def gradient(self, function):
        gradx = Derivative(function, x).doit()
        grady = Derivative(function, y).doit()
        return [gradx, grady]

    def secondDerivative(self, function):
        gradxx = Derivative(Derivative(function, x), x).doit()
        gradyy = Derivative(Derivative(function, y), y).doit()
        return [gradxx, gradyy]
        
    def update_weights(self, grads, velocity):

        velocity = np.multiply(self.momentum, velocity) + np.multiply(self.lr, grads)
        self.vars = np.subtract(self.vars, velocity)[0]
        return velocity

    def history_update(self, z, x, y, dx, dy):
        self.z_history.append(z)
        self.x_history.append(x)
        self.y_history.append(y)
        self.dx_history.append(dx)
        self.dy_history.append(dy)

    def func(self, variables):
        xi, yi = variables
        z = self.function.subs([(x,xi),(y,yi)])
        return z

    def grads(self, variables):
        
        xi, yi = variables
        dx = self.gradients[0].subs([(x,xi), (y,yi)])
        dy = self.gradients[1].subs([(x,xi), (y,yi)])
        grad = [dx, dy]
        return grad

    def getCurrentVelocity(self):
        return self.q.popleft()

    def saveVelocity(self, newVelocity):
        self.q.append(newVelocity)

    def createPath(self):
        self.x_history = np.array(self.x_history)
        self.y_history = np.array(self.y_history)
        #velocities = [np.array(self.dx_history)[:-1], np.array(self.dy_history)[:-1]]
        velocities = [np.array(self.dx_history), np.array(self.dy_history)]
        self.path = np.concatenate((np.expand_dims(self.x_history, 1), np.expand_dims(self.y_history, 1)), axis=1).T
        return velocities

    def checkGradients(self, tol, step):

        if self.oscillating[0] == False:
            val = self.dx_history[-tol+1]
            posDx = sum(self.dx_history[-tol:])
            negDx = sum(self.dx_history[-(2*tol)+1:-tol+1])
            print(posDx, negDx)
            
        if self.oscillating[1] == False:
            val = self.dy_history[-tol+1]
            posDy = sum(self.dy_history[-tol:])
            negDy = sum(self.dy_history[-(2*tol)+1:-tol+1])
            print(posDy, negDy)
            

    def train(self, max_iter):
        testingThreshold = 5
        zList = np.zeros(max_iter+1)
        for step in range(max_iter):
            currentVelocity = self.getCurrentVelocity()
            self.z = self.func(self.vars)
            zList[step+1] = self.z

            diff = np.abs(zList[step] - zList[step+1])

            
            self.grad = self.grads(self.vars)
            self.history_update(self.z, self.x, self.y, self.dx, self.dy)
            if len(self.dx_history) > 2*testingThreshold+1:
                self.checkGradients(testingThreshold, step)
            
            newVelo = self.update_weights(self.grad, currentVelocity)

            if (step+1) % 100 == 0:
                try:
                    print("steps: {}  z: {:.6f}  x: {:.5f}  y: {:.5f}  dx: {:.5f}  dy: {:.5f}".format(step+1, float(self.func(self.vars)), self.x, self.y, self.dx, self.dy))
                except Exception:
                    pass
            if np.abs(diff) < 1e-7 and step > 5:
                print("Enough convergence")
                print("steps: {}  z: {:.6f}  x: {:.5f}  y: {:.5f}".format(step+1, self.func(self.vars), self.x, self.y))
                self.z = self.func(self.vars)
                self.history_update(self.z, self.x, self.y, self.dx, self.dy)
                break
            self.saveVelocity(newVelo)
        
        velocities = self.createPath()
        return zList, step, velocities

    @property
    def x(self):
        return self.vars[0]

    @property
    def y(self):
        return self.vars[1]

    @property
    def dx(self):
        return self.grad[0]

    @property
    def dy(self):
        return self.grad[1]


class OptimiserWithVaryingDelay(Optimiser):
    def __init__(self, function, x_init=None, y_init=None, learning_rate=0.01, momentum=0.9, meanDelay=1):
        super().__init__(function, x_init, y_init, learning_rate, momentum, delay=None)
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
        #np.append(self.velocities, newVelocity)

    def update_weights(self, grads, velocity):

        velocity = np.multiply(self.momentum, velocity) + np.multiply(self.lr, grads)
        self.vars = np.subtract(self.vars, velocity)
        return velocity

    def createPath(self):
        self.x_history = np.array(self.x_history)
        self.y_history = np.array(self.y_history)
        velocities = [np.array(self.dx_history)[:-1], np.array(self.dy_history)[:-1]]
        self.path = np.concatenate((np.expand_dims(self.x_history, 1), np.expand_dims(self.y_history, 1)), axis=1).T
        return velocities
    

x = Symbol('x')
y = Symbol('y')
beale = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2   #1, 1.4
g = x**2 + y**2 + 4*x + 5*x
h = -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
goldsteinPrice = (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)) 
booth = (x+2*y-7)**2 + (2*x + y - 5)*2
sphere = x**2 + y**2
rosenbrock = (x-1)**2 + 10*(y-x**2)**2 #-1.4, -1
matyas = 0.26*(x**2+y**2)-0.48*x*y  #3, 3.5
levi = sin(3*pi*x)**2 + ((x-1)**2)*(1+sin(3*pi*y)**2) + ((y-1)**2)*(1+sin(2*pi*y))  #-3, 3
himmelblau = (x**2 + y - 11)**2 + (x+y**2-7)**2   #1, -3

opt = Optimiser(beale, 1, 1.2, momentum=0.8, delay=3)
zList, convergeIter, velocities = opt.train(1000)

Plot3D = Plot3D(50, beale, margin=4.5)
#Plot3D.plotMinima()
#Plot3D.contourPlotWithPath(opt.path)
Plot3D.plotContourWithMinima(opt.path, zList[:convergeIter+1], convergeIter, velocities)
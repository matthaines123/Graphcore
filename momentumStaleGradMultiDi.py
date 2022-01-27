from sympy import Symbol, Derivative
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from collections import deque

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
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.contour(self.x_mesh.astype(float), self.y_mesh.astype(float), self.z.astype(float), levels=np.logspace(-.5, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_xlim((self.x_min, self.x_max))
        ax.set_ylim((self.y_min, self.y_max))

        plt.show()


class Optimiser():
    def __init__(self, function, var_symbols, var_inits=None, learning_rate=0.01, momentum=0.9, delay=1):
        self.function = function
        self.var_symbols = var_symbols
        self.gradients = self.gradient(self.function)
        self.var_values = np.zeros([len(var_symbols)])
        
        scale = 3.0
        for var_index in range(len(var_symbols)):
            if var_inits is not None:
                self.var_values[var_index] = var_inits[var_index]
            else:
                self.var_values[var_inits] = np.random.uniform(low=-scale, high=scale)

        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = np.zeros([len(var_symbols)])

        self.q = deque()
        for _ in range(delay):
            self.q.append([self.velocity])

        self.var_values_history = [[] for var in range(len(var_symbols))]
        self.result_history = []

    def gradient(self, function):
        return [Derivative(function, symbol).doit() for symbol in self.var_symbols]

    def update_weights(self, grads, velocity):
        velocity = np.multiply(self.momentum, velocity) + np.multiply(self.lr, grads)
        self.var_values = np.subtract(self.var_values, velocity)[0]
        return velocity

    def history_update(self):
        self.result_history.append(self.result)
        for var_index, value in enumerate(self.var_values):
            self.var_values_history[var_index].append(value)

    def func(self, pri=False):
        var_val_pairs = []
        for symbol, var_value in zip(self.var_symbols, self.var_values):
            var_val_pairs.append((symbol, var_value))
        result = self.function.subs(var_val_pairs)
        return result

    def grads(self):
        var_val_pairs = []
        for symbol, var_value in zip(self.var_symbols, self.var_values):
            var_val_pairs.append((symbol, var_value))

        all_gradients = []
        for gradient in self.gradients:
            dn = gradient.subs(var_val_pairs)
            all_gradients.append(dn)
        return all_gradients

    def print_var_values(self):
        value_text = 'Variable Values:\n'
        for index, val in enumerate(self.var_values):
            value_text += 'x{}: {} '.format(index + 1, val)
        print(value_text)

    def print_diff_values(self):
        diff_text = 'Gradients:\n'
        for index, grad in enumerate(self.grad):
            diff_text += 'dx{}: {} '.format(index + 1, grad)
        print(diff_text)

    def train(self, max_iter):
        resultList = np.zeros(max_iter+1)
        for step in range(max_iter):            
            currentVelocity = self.q.popleft()
            self.result = self.func()
            resultList[step+1] = self.result
            
            diff = np.abs(resultList[step] - resultList[step+1])

            self.history_update()
            self.grad = self.grads()
            newVelo = self.update_weights(self.grad, currentVelocity)

            self.result = self.func()
            if (step+1) % 100 == 0:
                print('\nSteps: {}  Function Value: {:.6f}'.format(step+1, self.result))
                self.print_var_values()
                self.print_diff_values()

            if np.abs(diff) < 1e-7 and step > 5:
                print('\nEnough Convergence')
                print('Steps: {}  Function Value: {:.6f}'.format(step+1, self.result))
                self.print_var_values()
                self.print_diff_values()
                break
            self.q.append(newVelo)

        self.var_values_history = np.array(self.var_values_history)
        # If there are exactly three dimensions
        if len(self.var_values_history) == 2:
            self.path = np.concatenate((np.expand_dims(self.var_values_history[0], 1), np.expand_dims(self.var_values_history[1], 1)), axis=1).T

def main():
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    
    f = (1.5 - x1 + x1*x2)**2*x3 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2
    #g = x**2 + y**2 + 4*x + 5*x

    var_symbols = [x1, x2, x3]
    var_inits = [1.0, 1.0, 1.0]

    opt = Optimiser(f, var_symbols, var_inits, momentum=0.8, delay=2)
    opt.train(1000)

    if len(var_symbols) == 2: 
        Plot3D = Plot3D(20, f, margin=4.5)
        Plot3D.contourPlotWithPath(opt.path)

if __name__ == '__main__':
    main()
    

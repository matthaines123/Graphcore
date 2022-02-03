from sympy import Derivative
import numpy as np
from collections import deque


class Optimiser():
    def __init__(self, function, varSymbols, varInits, **kwargs):
        self.function = function
        self.varSymbols = varSymbols
        self.gradients = self.gradient(self.function)

        self.funcValue = None
        self.varValues = np.zeros([len(varSymbols)])
        self.diffValues = np.zeros([len(varSymbols)])

        scale = 3.0
        for varIndex in range(len(varSymbols)):
            self.varValues[varIndex] = varInits[varIndex]
        
        self.momentumDecay = False
        self.learningRateDecay = False

        self.tol = kwargs['tol']
        self.lr = np.array([kwargs["learning_rate"]] * len(self.varSymbols))
        self.momentumVec = np.array([kwargs["momentum_rate"]] * len(self.varSymbols))
        self.velocity = np.zeros([len(varSymbols)])
        if kwargs['delay'] != None:
            self.q = deque()
            for _ in range(kwargs['delay']):
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
            
            if self.momentumDecay:
                self.variableMomentum(step)
            if self.learningRateDecay:
                self.variableLR(step)

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
from BaseOptimiser import Optimiser
import numpy as np

class OptimiserWithMomentumDecay(Optimiser):
    def __init__(self, function, varSymbols, varInits=None, tol=1e-7, learning_rate=0.01, momentum=0.9, variableMomentumScalar=None, delay=1):
        super().__init__(function, varSymbols, varInits, tol, learning_rate, momentum, variableMomentumScalar, delay=delay)
        self.variableMomentumScalar = variableMomentumScalar

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
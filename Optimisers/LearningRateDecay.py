from BaseOptimiser import Optimiser
import numpy as np

class OptimiserWithLearningRateDecay(Optimiser):
    def __init__(self, function, varSymbols, varInits=None, **kwargs):
        super().__init__(function, varSymbols, varInits, **kwargs)
        self.learningRateScalar = kwargs['learning_rate_scalar']
        self.learningRateDecay = True

    def variableLR(self, step):
        for index, dnHistory in enumerate(self.diffValuesHistory):
            dn1 = self.diffValuesHistory[index][step]
            dn2 = self.diffValuesHistory[index][step - 1]
            if dn1 > 0 and dn2 < 0:
                self.lr *= self.learningRateScalar
                return
            elif dn1 < 0 and dn2 > 0:
                self.lr *= self.learningRateScalar
                return
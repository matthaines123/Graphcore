from BaseOptimiser import Optimiser
import numpy as np

class OptimiserWithMomentumDecay(Optimiser):
    def __init__(self, function, varSymbols, varInits=None, **kwargs):
        super().__init__(function, varSymbols, varInits, **kwargs)
        self.variableMomentumScalar = kwargs['variable_momentum_scalar']
        self.momentumDecay = True
        self.oscillationFrequency = 0

    def variableMomentum(self, step):
        for index, dnHistory in enumerate(self.diffValuesHistory):
            dn1 = dnHistory[step]
            dn2 = dnHistory[step - 1]
            if dn1 > 0 and dn2 < 0:
                self.momentumVec[index] *= self.variableMomentumScalar
            elif dn1 < 0 and dn2 > 0:
                self.momentumVec[index] *= self.variableMomentumScalar


   

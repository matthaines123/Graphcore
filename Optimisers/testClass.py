from MomentumDecay import OptimiserWithMomentumDecay
from LearningRateDecay import OptimiserWithLearningRateDecay


class LearningRateAndMomentumDecay(OptimiserWithLearningRateDecay, OptimiserWithMomentumDecay):
    def __init__(self, function, varSymbols, varInits=None, **kwargs):
        super().__init__(function, varSymbols, varInits, **kwargs)
        
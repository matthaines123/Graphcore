from BaseOptimiser import Optimiser
import numpy as np
from sympy import exp


class OptimiserWithLearningRateDecay(Optimiser):
    def __init__(self, function, varSymbols, varInits=None, **kwargs):
        super().__init__(function, varSymbols, varInits, **kwargs)
        self.learningRateScalar = kwargs['learning_rate_scalar']
        self.delay = kwargs['delay']
        self.learningRateDecay = True
        self.oscillations = []
        self.iterateRange = 10
        self.lrPoint = 0.75

        #self.init_lr = np.zeros([len(varSymbols)])
        for dim in range(len(varSymbols)):
            self.oscillations.append([])

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


    def variableLR(self, step):
        for index, dnHistory in enumerate(self.diffValuesHistory):
            isOscillating = self.checkOscillating(step, dnHistory)
            self.oscillations[index].append(isOscillating)
            '''if step == 0:
                self.init_lr[index] = self.momentumVec[index]'''
            if isOscillating:
                oscillationFreq = self.getOscillatingFrequency(index)
                #Only change the momentum if the oscillation frequency is over a threshold
                    #Updating the momentum
                self.lr[index] = self.updateLR(self.lr[index], oscillationFreq)
            

            '''elif (self.init_moment[index]  > self.momentumVec[index] ) & (isOscillating == False ) :
                
                #self.momentumVec[index] = self.updateMomentum_method1(self.momentumVec[index] , self.delay)
                self.momentumVec[index] = self.updateMomentum_method2(self.momentumVec[index] , step , self.delay, maxIter)'''
             

    def updateLR(self, currentLR, oscillationFreq): 
        pointOfInflection = self.iterateRange*self.lrPoint
        sigmoidFactor = (1-self.learningRateScalar/(1+exp(-(oscillationFreq-pointOfInflection))))
        return sigmoidFactor*currentLR
    
    #constant small ratio of deacy. can be ignore
    def updateMomentum_method1(self, currentMomentum, decay):
        currentMomentum = (1-self.delay*0.003)*currentMomentum
        return currentMomentum


    #instead of decrease the momentum increase momentum in ratio that ratio icrease over time if dont oscillation
    #If it dont oscillate, it's likely that it has been denoise , so we can increase the momentum in order to reduce the iteration
    def updateMomentum_method2(self, currentMomentum, step, delay, maxIter):
        currentMomentum = currentMomentum * (1+exp(-maxIter*1.2/(step+1)))
        return currentMomentum

    def checkOscillating(self, step, dnHistory):
        dn1 = dnHistory[step]
        dn2 = dnHistory[step - 1]
        #Checking if the gradient has changed direction
        if dn1 * dn2 < 0:
            return True
        else:
            return False

    def getOscillatingFrequency(self, dim):
        
        #Returns the number of oscillations in the last ten iterates
        return sum(self.oscillations[dim][-self.iterateRange:])
    

            

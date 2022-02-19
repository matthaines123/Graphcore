from BaseFunctions import *
from MomentumDecay import *
from VaryingDelay import *
from Plot3D import *
from LearningRateMomentumDecay import *
from BaseOptimiser import Optimiser
import matplotlib.pyplot as plt

def main():
    # Selecting the function
    function, varSymbols = BaseFunctions().beale()

    # Initial values
    varInits = [1, 1]

    # Adding function & initial conditions to optimiser
    
    opt = OptimiserWithMomentumDecay(function, varSymbols, varInits, tol=1e-5, learning_rate=0.02, learning_rate_scalar=0.05, momentum_rate=0.45, variable_momentum_scalar=0.99, delay=2)
    funcValues, convergeIter, velocities, path = opt.train(1000)
    
    '''newOscillations = [[],[]]
    current = 0
    for i in opt.oscillationFreq[0]:
        
        if i < current:
            newOscillations[0].append(current)
        else:
            newOscillations[0].append(i)
            current = i
    current = 0
    for i in opt.oscillationFreq[1]:
        
        if i < current:
            newOscillations[1].append(current)
        else:
            newOscillations[1].append(i)
            current = i


    plt.plot(range(0, convergeIter+1), newOscillations[0],label='x')
    plt.plot(range(0, convergeIter+1), newOscillations[1],label='y')
    plt.xlabel('Iteration')
    plt.ylabel('Oscillation Frequency')
    plt.legend()
    plt.show()'''

    # Plotting results only if the function is in 3-Dimensions
    if len(varSymbols) == 2:
        plot3D = Plot3D(50, function, margin=4.5)
        plot3D.contourPlot(path, funcValues[:convergeIter+1], convergeIter, velocities)

        # Other types of plots
        
        #plot3D.plotMinima()
        #plot3D.contourPlotWithPath(path)
    
if __name__ =="__main__":
    main()

#.yaml config

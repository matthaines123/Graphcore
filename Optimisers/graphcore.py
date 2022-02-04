from BaseFunctions import *
from MomentumDecay import *
from VaryingDelay import *
from Plot3D import *
from LearningRateMomentumDecay import *

def main():
    # Selecting the function
    function, varSymbols = BaseFunctions().beale()

    # Initial values
    varInits = [1, 1.2]

    # Adding function & initial conditions to optimiser
    opt = OptimiserWithMomentumDecay(function, varSymbols, varInits, tol=1e-7, learning_rate=0.02, momentum_rate=0.8, variable_momentum_scalar=0.1, delay=2)
    funcValues, convergeIter, velocities, path = opt.train(1000)

    # Plotting results only if the function is in 3-Dimensions
    if len(varSymbols):
        plot3D = Plot3D(50, function, margin=4.5)
        plot3D.plotContourWithMinima(path, funcValues[:convergeIter+1], convergeIter, velocities)

        # Other types of plots
        
        #plot3D.plotMinima()
        #plot3D.contourPlotWithPath(path)
    
if __name__ =="__main__":
    main()

#.yaml config

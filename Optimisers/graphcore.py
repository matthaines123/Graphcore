from BaseFunctions import *
from MomentumDecay import *
from VaryingDelay import *
from Plot3D import *

def main():
    # Selecting the function
    beale, varSymbols = BaseFunctions().beale()

    # Initial values
    varInits = [1, 1.2]

    # Adding function & initial conditions to optimiser
    opt = OptimiserWithMomentumDecay(beale, varSymbols, varInits, tol=1e-7, learning_rate=0.02, momentum=0.8, variableMomentumScalar=0.95, delay=30)
    funcValues, convergeIter, velocities, path = opt.train(1000)

    # Plotting results only if the function is in 3-Dimensions
    if len(varSymbols):
        plot3D = Plot3D(50, beale, margin=4.5)
        plot3D.plotContourWithMinima(path, funcValues[:convergeIter+1], convergeIter, velocities)

        # Other types of plots
        """Plot3D.plotMinima()
        #Plot3D.contourPlotWithPath(path)"""
    
if __name__ =="__main__":
    main()
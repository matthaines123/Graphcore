from BaseFunctions import *
from MomentumDecay import *
from VaryingDelay import *
from Plot3D import *
from LearningRateMomentumDecay import *

def main():
    learningRateScalars = np.linspace(0.1, 0.95, num=10)
    momentumRateScalars = np.linspace(0.1, 0.95, num=10)
    converges = []
    for lrscalar in learningRateScalars:
        for momentumRate in momentumRateScalars:
            print(lrscalar)
        # Selecting the function
            function, varSymbols = BaseFunctions().beale()

            # Initial values
            varInits = [1, 1.2]

            # Adding function & initial conditions to optimiser
            opt = OptimiserWithMomentumDecay(function, varSymbols, varInits, tol=1e-5, learning_rate=0.02, learning_rate_scalar=0.9, momentum_rate=momentumRate, variable_momentum_scalar=lrscalar, delay=1)
            try:
                funcValues, convergeIter, velocities, path = opt.train(10000)
                converges.append(convergeIter)
            except RuntimeWarning: 
                learningRateScalars =  np.delete(learningRateScalars, np.where(learningRateScalars == lrscalar))
                

        # Plotting results only if the function is in 3-Dimensions
        

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(learningRateScalars, momentumRateScalars)
    ax.contour3D(X, Y, converges, 50, cmap='binary')
    plt.show()
        
if __name__ =="__main__":
    main()

#.yaml config

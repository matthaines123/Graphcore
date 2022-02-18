from BaseFunctions import *
from MomentumDecay import *
from VaryingDelay import *
from Plot3D import *
from LearningRateMomentumDecay import *
import warnings
warnings.filterwarnings('ignore')
 
 
def main():
    learningRateScalars = np.linspace(0.4, 0.90, num=20)
    momentumRateScalars = np.linspace(0.1, 0.90, num=20)
    converges = []
    minimum = 0
    errors = []
    for lrscalar in learningRateScalars:
        for momentumRate in momentumRateScalars:
            print(lrscalar, momentumRate)
        # Selecting the function
            function, varSymbols = BaseFunctions().beale()
 
            # Initial values
            varInits = [1, 1.2]
 
            # Adding function & initial conditions to optimiser
            opt = OptimiserWithMomentumDecay(function, varSymbols, varInits, tol=1e-5, learning_rate=0.02, learning_rate_scalar=0.9, momentum_rate=momentumRate, variable_momentum_scalar=lrscalar, delay=2)
            try:
                funcValues, convergeIter, velocities, path = opt.train(10000)
                error = abs(funcValues - minimum)
                converges.append(convergeIter)
                errors.append(error)
            except RuntimeWarning:
                converges.append(10000)
                errors.append(1)
               
 
        # Plotting results only if the function is in 3-Dimensions
       
 
    fig = plt.figure()
    X, Y = np.meshgrid(learningRateScalars, momentumRateScalars)
    convergesMatrix = np.reshape(converges, (-1,len(learningRateScalars)))
    
    errorsMatrix = np.reshape(errors, (-1, len(learningRateScalars)))

 


    ax = fig.add_subplot(2,2,1,projection='3d')
 
    ax.plot_surface(X, Y, errorsMatrix.astype(float), rstride=1, cstride=1, edgecolor='none', cmap='winter')
    ax.set_xlabel('Variable Momentum Scalar')
    ax.set_ylabel('Momentum Rate')
    ax.set_zlabel('Absolute Error in Minima')
 
    ax = fig.add_subplot(2,2,2,projection='3d')
 
    ax.plot_surface(X, Y, convergesMatrix, rstride=1, cstride=1, edgecolor='none', cmap='winter')
    ax.set_xlabel('Variable Momentum Scalar')
    ax.set_ylabel('Momentum Rate')
    ax.set_zlabel('Iterations to Convergence')
 
 
    plt.show()
       
if __name__ =="__main__":
    main()
 
#.yaml config
 


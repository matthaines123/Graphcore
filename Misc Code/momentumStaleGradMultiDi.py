from Plot3D import Plot3D
from BaseOptimiser import Optimiser
from VaryingDelay import OptimiserWithVaryingDelay
from MomentumDecay import OptimiserWithMomentumDecay
from LearningRateDecay import OptimiserWithLearningRateDecay
from LearningRateMomentumDecay import LearningRateAndMomentumDecay

from sympy import Symbol, Derivative
import numpy as np
import time
import os
import math

from sympy import sin, cos, exp, sqrt
from numpy import pi, e
from decimal import Decimal, DecimalException


x = Symbol('x')
y = Symbol('y')
varSymbols = [x, y]

""" Functions """
beale = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2   #1, 1.4
g = x**2 + y**2 + 4*x + 5*x
h = -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
goldsteinPrice = (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)) 
booth = (x+2*y-7)**2 + (2*x + y - 5)*2
sphere = x**2 + y**2
rosenbrock = (x-1)**2 + 10*(y-x**2)**2 #-1.4, -1
matyas = 0.26*(x**2+y**2)-0.48*x*y  #3, 3.5
levi = sin(3*pi*x)**2 + ((x-1)**2)*(1+sin(3*pi*y)**2) + ((y-1)**2)*(1+sin(2*pi*y))  #-3, 3
himmelblau = (x**2 + y - 11)**2 + (x+y**2-7)**2 #1, -3
""""""

varInits = [1, 1.4]
opt = LearningRateAndMomentumDecay(beale, varSymbols, varInits, tol=1e-7, learning_rate=0.01, momentum_rate=0.8, learning_rate_scalar=0.95, variable_momentum_scalar=0.95, delay=3,)
zList, convergeIter, velocities, path = opt.train(1000)

Plot3D = Plot3D(50, beale, margin=4.5)
#Plot3D.plotMinima()
#Plot3D.contourPlotWithPath(opt.path)
Plot3D.plotContourWithMinima(path, zList[:convergeIter+1], convergeIter, velocities)

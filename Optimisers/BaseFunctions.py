from sympy import Symbol
from sympy import sin, cos, exp, sqrt
from numpy import pi, e


class BaseFunctions():
    def __init__(self):
        self.x = Symbol('x')
        self.y = Symbol('y')
        
    def beale(self):
        x = self.x
        y = self.y
        function = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
        return function, [x, y]

    def goldsteinPrice(self):
        x = self.x
        y = self.y
        function = ((1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2)) *
                    (30+(2*x-3*y)**2 *
                    (18-32*x+12*x**2+48*y-36*x*y+27*y**2)))
        return function, [x, y]

    def booth(self):
        x = self.x
        y = self.y
        function = (x+2*y-7)**2 + (2*x + y - 5)*2
        return function, [x, y]

    def sphere(self):
        x = self.x
        y = self.y
        function = x**2 + y**2
        return function, [x, y]

    def rosenbrock(self):
        x = self.x
        y = self.y
        function = (x-1)**2 + 10*(y-x**2)**2
        return function, [x, y]

    def matyas(self):
        x = self.x
        y = self.y
        function = 0.26*(x**2+y**2)-0.48*x*y
        return function, [x, y]

    def levi(self):
        x = self.x
        y = self.y
        function = (sin(3*pi*x)**2 +
                    ((x-1)**2)*(1+sin(3*pi*y)**2) +
                    ((y-1)**2)*(1+sin(2*pi*y)))
        return function, [x, y]
        
    def himmelblau(self):
        x = self.x
        y = self.y
        function = ((x**2 + y - 11)**2 +
                    (x+y**2-7)**2)
        return function, [x, y]
    

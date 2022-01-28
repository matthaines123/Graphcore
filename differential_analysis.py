
import numpy as np
import matplotlib.pyplot as plt

darray = np.loadtxt('differentials.txt')
[s1,s2] = darray.shape

for dim in range(s2):

   plt.semilogy(darray[0:s1,dim])
   plt.ylabel('differential value')
   plt.show()
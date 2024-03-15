# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:40:35 2024

@author: Han Ruiyu

Experiments of distributions in R^d, can be Gaussian, uniform, or Pareto
"""

from max_sliced_functions import *
from time import time
import matplotlib.pyplot as plt


number_of_points=[50, 100, 250, 500, 1000]
max_sw_distance=[0,0,0,0,0]


d=10
for i in range(5):
    start = time()
    num=number_of_points[i]
    
    max_sw_distance[i]=expected_max_swdistance_empirical('Gaussian',2,d,num,2)
    #max_sw_distance[i]=expected_max_swdistance_gaussian(2,d, num, 2)
    stop = time()
    
    print("Runtime for "+str(num) +" samples:"+ str((stop - start) / 60.0) + " min.\n")

# if want to plot log-log
#plt.loglog(number_of_points, max_sw_distance)

# plotting parameters
plt.plot(number_of_points, max_sw_distance)
plt.title('${\mu}=\mathcal{N}(0,4I_{10})$')
plt.xlabel('number of sample points $n$')
plt.ylabel('$\mathbb{E}\overline{W}_2(\mu_n,\mu)$')
plt.grid()



plt.show()

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
max_sw_distance_1=[0,0,0,0,0]
max_sw_distance_2=[0,0,0,0,0]
max_sw_distance_3=[0,0,0,0,0]



for i in range(5):
    start = time()
    num=number_of_points[i]
    
    #max_sw_distance_1[i]=expected_max_swdistance_empirical_covar_Gaussian(num,2)
    
    #max_sw_distance_2[i]=expected_max_swdistance_empirical('Gaussian', 8, 4, num, 2)
    #max_sw_distance_2[i]=expected_max_swdistance_empirical('Gaussian', 8, 4, num, 2)
    #max_sw_distance_3[i]=expected_max_swdistance_empirical('Gaussian', 8, 8, num, 2)
    max_sw_distance_1[i]=expected_max_swdistance_gaussian(1,2, num, 2)
    max_sw_distance_2[i]=expected_max_swdistance_gaussian(1,4, num, 2)
    max_sw_distance_3[i]=expected_max_swdistance_gaussian(1,8, num, 2)
    stop = time()
    
    print("Runtime for "+str(num) +" samples:"+ str((stop - start) / 60.0) + " min.\n")

# if want to plot log-log
#plt.loglog(number_of_points, max_sw_distance)

# plotting parameters
#plt.plot(number_of_points, max_sw_distance)
plt.plot(number_of_points, max_sw_distance_1)
plt.plot(number_of_points, max_sw_distance_2)
plt.plot(number_of_points, max_sw_distance_3)
plt.title('${\mu}=\mathcal{N}(0,I_d)$')
plt.xlabel('number of sample points $n$')
plt.ylabel('$\mathbb{E}\overline{W}_2(\mu_n,\mu)$')
plt.legend(['d=2','d=4','d=8'])
plt.grid()



plt.show()
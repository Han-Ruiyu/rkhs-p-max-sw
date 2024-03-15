# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:15:36 2024

@author: Han Ruiyu

Experiments of the rkhs embedding part
the rkhs has Gaussian kernel
the sampled distribution can be 1d uniform distribution
or 1d Gaussian
"""

from max_sliced_functions import *
from time import time
import matplotlib.pyplot as plt

    
number_of_points=[50,100,250, 500,1000]
max_sw_distance=[0,0,0,0,0]


for i in range(5):
    start = time()
    num=number_of_points[i]
    
    # number of times of monte-carlo
    iterate=100
    
    max_sw_distance[i]=expected_max_swdistance_empirical_embed_into_gaussian_rkhs('Gaussian',2,num,iterate)
    stop = time()
    
    print("Runtime for "+str(num) +" samples:"+ str((stop - start) / 60.0) + " min.\n")

# if want to plot log-log
#plt.loglog(number_of_points, max_sw_distance)
#plt.title('log-log plot ${\mu}=$')

plt.plot(number_of_points, max_sw_distance)
plt.title('${\mu}=\mathcal{N}(0,1)$')
plt.xlabel('number of sample points $n$')
plt.ylabel('$\mathbb{E}\overline{W}_2(\Phi_{\#}(\mu_n),\Phi_{\#}(\mu))$')
plt.grid()



plt.show()

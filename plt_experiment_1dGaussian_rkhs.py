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
max_sw_distance_d10=[0,0,0,0,0]
max_sw_distance_d20=[0,0,0,0,0]
max_sw_distance_d30=[0,0,0,0,0]
#max_sw_distance_12=[0,0,0,0,0]
#max_sw_distance_11=[0,0,0,0,0]
#max_sw_distance_1squareroot2=[0,0,0,0,0]


for i in range(5):
    start = time()
    num=number_of_points[i]
    
    
    #max_sw_distance_d10[i]=expected_max_swdistance_empirical_embed_into_gaussian_rkhs('Gaussian',2,num,1,10,200)
    
    max_sw_distance_d10[i]=expected_max_swdistance_empirical_embed_into_gaussian_rkhs('Gaussian',2,num,1,10,150)
    max_sw_distance_d20[i]=expected_max_swdistance_empirical_embed_into_gaussian_rkhs('Gaussian',2,num,1,20,150)
    max_sw_distance_d30[i]=expected_max_swdistance_empirical_embed_into_gaussian_rkhs('Gaussian',2,num,1,30,150)
    #max_sw_distance_d20[i]=expected_max_swdistance_empirical_embed_into_gaussian_rkhs('Gaussian',2,num,1,30)
    #max_sw_distance_1squareroot2[i]=expected_max_swdistance_empirical_embed_into_gaussian_rkhs('Gaussian',2,num,1/np.sqrt(2))
    stop = time()
    
    print("Runtime for "+str(num) +" samples:"+ str((stop - start) / 60.0) + " min.\n")

# if want to plot log-log
#plt.loglog(number_of_points, max_sw_distance)
#plt.title('log-log plot ${\mu}=\mathcal{N}(0,9)$')

plt.plot(number_of_points,max_sw_distance_d10)
plt.plot(number_of_points, max_sw_distance_d20)
plt.plot(number_of_points,max_sw_distance_d30)
plt.plot(number_of_points, np.power(np.log(number_of_points),11/12)*np.power(number_of_points,-1/4))
plt.title('${\mu}=v=\mathcal{N}(0,1)$')
plt.xlabel('number of sample points $n$',fontsize=18)

plt.ylabel('$\mathbb{E}\overline{W}_2(\Phi_{\#}(\mu_n),\Phi_{\#}(\mu))$',fontsize=16)
plt.legend(['$d_{test}=10$','$d_{test}=20$','$d_{test}=30$', '$C*n^{-1/4}(\log n)^{11/12}$'],loc='center right',bbox_to_anchor=(1.45, 0.8))
plt.grid()

plt.show()
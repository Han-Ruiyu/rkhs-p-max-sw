#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 12:47:21 2025

@author: ruiyuhan
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time


def phidistance(x,y,d):
    m=np.size(x)
    n=np.size(y)

    vecx=[np.power(x,i+1) for i in range(d)]
    vecy=[np.power(y,i+1) for i in range(d)]

    xempirical=np.sum(vecx,axis=1)
    yempirical=np.sum(vecy,axis=1)

    difference=xempirical/m - yempirical/n
    
    return difference



def Gaussiankerneldistance(x,y,sigma=0.5):
    m=np.size(x)
    n=np.size(y)
    
    [x1,x2]=np.meshgrid(x,x)
    matrixxx=np.exp(-(x1-x2)**2/sigma)
    
    
    [y1,y2]=np.meshgrid(y,y)
    matrixyy=np.exp(-(y1-y2)**2/sigma)
    
    
    [x3,y3]=np.meshgrid(x,y)
    matrixxy=np.exp(-(x3-y3)**2/sigma)
    

    
    
    return np.sum( matrixxx)/(m**2) + np.sum( matrixyy)/(n**2)-2*np.sum(matrixxy)/(m*n)


# compute the sample standard deviation
def standardeviation(v):
    n=np.size(v)
    samplevec=v[0:n-1]-v[-1]
    samplesum=np.sum(samplevec**2)
    return np.sqrt(samplesum/(n-1))

# compute the MMD threshold
def MMDthreshold(numberlist,alpha=0.05,K=1):
    numberlist=np.array(numberlist)
    return np.sqrt(2*K/numberlist)*(1+np.sqrt(2*np.log(1/alpha)))


# parameters
loc,scale=0,1
mu,sigma=0,np.sqrt(2)
dim=[1,2,3,4]

testnumber=[50,100,250,500, 1000,2000]
# 50 100 500 1000 5000 10000
testresulttotal=[]
n_montecarlo=1200

for d in dim:
    testresult=[]
    for num in testnumber:
        s=0
        for i in range(n_montecarlo):
            X=np.random.normal(mu,sigma,num)
            Y= np.random.laplace(loc, scale, num)
            #Y=np.random.uniform(-1,1,num)
            vec=phidistance(X,Y,d)
            s+=np.sqrt(np.sum(vec**2))
        
    
        testresult.append(s/n_montecarlo)
    plt.plot(testnumber, testresult) 
    testresulttotal.append(testresult)

# the number here is computed manually
plt.plot(testnumber, [12,12,12,12,12,12])


# plot
plt.xlabel('number of samples')
plt.ylabel('$E[T(\~X,\~Y)]$')
plt.title('$\~\mu=Lap(0,1)$, $\~v=\mathcal{N}(0,2)$')
plt.legend(['d=1','d=2','d=3','d=4','$T(\~X,\~Y)=12$'],bbox_to_anchor = (1.05,1), loc = "upper left")
plt.savefig("test_lap_gaus.pdf", format = 'pdf', bbox_inches = 'tight')

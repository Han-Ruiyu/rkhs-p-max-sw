# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:59:36 2025

@author: Han Ruiyu
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
    

    
    
    return np.sqrt(np.sum( matrixxx)/(m**2) + np.sum( matrixyy)/(n**2)-2*np.sum(matrixxy)/(m*n))


def standardeviation(v):
    n=np.size(v)
    samplevec=v[0:n-1]-v[-1]
    samplesum=np.sum(samplevec**2)
    return np.sqrt(samplesum/(n-1))

def MMDthreshold(numberlist,alpha=0.05,K=1):
    numberlist=np.array(numberlist)
    return np.sqrt(2*K/numberlist)*(1+np.sqrt(2*np.log(1/alpha)))



mu1,sigma1=0,np.sqrt(2)
mu2,sigma2=0,np.sqrt(0.1)


testnumber=[50,100,250, 500,1000]
# 50 100 500 1000 5000 10000
#testresulttotal=[]
n_montecarlo=1000

# document the true value
#N(0,2)N(0,0.1) standard deviation [0.09604488235903941, 0.06773428004785285, 0.042769759324117744, 0.030226371028023377, 0.021367452843053067, 0.015107010754177216]
#N(0,2)N(0,0.1) mean[0.35580571213209544, 0.34568462830745805, 0.3396119780126756, 0.33758776124774814, 0.33657565286528446, 0.3360695986740526]


#N(0,1)N(3,1) standard deviation [0.06777570801185138, 0.04772375608906079, 0.030104623499838726, 0.0212684015757085, 0.01503235437050845, 0.010627112761569702]
#N(0,1)N(3,1) mean[0.6031098111755916, 0.5897764778422583, 0.5817764778422583, 0.5791098111755915, 0.5777764778422583, 0.5771098111755915]

#N(0,1)N(4,3) standard deviation [0.05844917418071705, 0.04103903870400102, 0.025842019566439525, 0.018245991815968945, 0.01289224398175354, 0.009112783265455475]
#N(0,1)N(4,3) mean [0.4888235464071498, 0.4741568797404832, 0.4653568797404832, 0.46242354640714983, 0.4609568797404832, 0.46022354640714974]

testresultvector=[]


for ind in range(5):
    num=testnumber[ind]
    start=time()
    testvector=[]
    for i in range(n_montecarlo):
        X=np.random.normal(mu1,sigma1,num)
        Y= np.random.normal(mu2, sigma2, num)
        g=Gaussiankerneldistance(X,Y)
        #difference=np.abs(g-meanarray[ind])
        testvector.append(g)
        
        
        
    stop=time()
    s=np.sum(testvector)
    testvector.append(s/n_montecarlo)
    testvector.append(standardeviation(testvector))
    testresultvector.append(testvector)
    print("Runtime for "+str(num) +" samples:"+ str((stop - start) / 60.0) + " min.\n")
    
    
testresultvector=np.array(testresultvector)

testmean=testresultvector[:,-2]
testdeviation=testresultvector[:,-1]
#plt.plot(testnumber, testmean, color='blue')
plt.plot(testnumber, [0.5792, 0.5792,0.5792,0.5792,0.5792],color='red')
plt.plot(testnumber, MMDthreshold(testnumber),color=(0.5,0,1))
plt.errorbar(testnumber,testmean, yerr=[3*testdeviation,3*testdeviation],color='green')
plt.xlabel('number of samples')
plt.ylabel('$T(\~X,\~Y)$')
plt.title('$\~\mu=\mathcal{N}(0,2)$, $v=\~\mathcal{N}(0,0.1)$')
plt.legend(['True value',
            'MMD threshold','3$\sigma$ interval'],bbox_to_anchor = (1.05,1), loc = "upper left")
plt.savefig("N(0,2)N(0,0.1).pdf", format = 'pdf', bbox_inches = 'tight')

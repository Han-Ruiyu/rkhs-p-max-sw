# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 16:22:13 2024

@author: Han Ruiyu

compute the p power of
p-max-sliced Wasserstein distance between empirical measre
and the true measure

The true measure could be set as Gaussian

Or compute the distance between two empirical meausures of
uniform distribution

Or compute the distance between two empirical measures of
Pareto distribution

Or compute the distance between two embedded empirical measuress
"""



import numpy as np
import scipy



   
def max_swdistance_empirical(I0,I1,N, p, L=2500):
    '''
    Given two d-dimensional measures I0, I1, compute the p-max-sliced 
    Wasserstein distance between them
    
    Use the fact that normalized vectors of Gaussian variables is 
    uniformly distributed on a sphere
    '''
    
    
    d = I0.shape[1]
    
    # number of direction is related to dimension d, take 180
    # this may not be a good choice
    
    
    
    # generate a random vector on sphere, compute the wasserstein
    # distance of the pushforward measures, select the biggest 
    
    theta_mat=np.random.randn(d,L)
    theta_mat= theta_mat/np.sqrt(np.sum(theta_mat ** 2,axis=0))
    
    # compute the pushforward measures
    pushforward_points0=np.dot(I0,theta_mat)
    pushforward_points1=np.dot(I1,theta_mat)
    
    # sort the points to obtain the optimal coupling
    index_0=np.argsort(pushforward_points0,axis=0)
    index_1=np.argsort(pushforward_points1,axis=0)
        
    points_0=np.take_along_axis(pushforward_points0,index_0,axis=0)
    points_1=np.take_along_axis(pushforward_points1,index_1,axis=0)
    
    # p-Wasserstein distance between pushforward measures
    sw_piece=np.sum(np.power(abs(points_0-points_1),p),axis=0)
        
    max_sw=np.max(sw_piece)

    # take 1/p power
    return np.power(max_sw/N,1/p)






def sample_from_Pareto(uniform_rand, a):
    '''
    
    Use inversion method to generate sample from Pareto distributions,
    a is the shape parameter of Pareto distribution

    '''
    sample_points=1/np.power(1-uniform_rand,1/a)
    
    return sample_points





def expected_max_swdistance_empirical(distribution, sigma, d, N, p, n_montecarlo=100):
    '''
    Compute the expection of p-max-sliced Wasserstein distance
    between two empirical measures
    
    

    Parameters
    ----------
    distribution : string
        can be 'uniform', 'Gaussian', 'Pareta'
    sigma : positive real number
        DESCRIPTION: 'uniform': sample from U([-sigma,sigma])
                     'Gaussian': sample from N(0,sigma^2)
                     'Pareto': shape parameter of Pareto distribution,
                               the code behaves unstable if a is small
                               
    d : integer
        dimension of target distribution
    N : integer
        number of samples
    p : p>=1
        p-max-sliced Wasserstein distance
        
    n_montecarlo : TYPE
        times of montecarlo

    Returns
    -------
    positive real number
        DESCRIPTION. expection of p-max-sliced Wasserstein distance
        between two empirical measures

    '''
    
    
    
    
    s=0
    
    if distribution=='uniform':
        for n in range(n_montecarlo):
            points1=sigma*(-1+2*np.random.rand(N,d))
        
            
            points2=sigma*(-1+2*np.random.rand(N,d))
            
            max_sw_temp=max_swdistance_empirical(points1,points2,N, p)
            s+= max_sw_temp
        
        
    if distribution=='Gaussian':
        
        for n in range(n_montecarlo):
            points1=sigma*np.random.randn(N,d)
            
            points2=sigma*np.random.randn(N,d)
            
            max_sw_temp=max_swdistance_empirical(points1,points2,N, p)
            s+= max_sw_temp 
            
    if distribution=='Pareto':
        for n in range(n_montecarlo):
            
            points1=sample_from_Pareto(np.random.rand(N,d),sigma)
            points2=sample_from_Pareto(np.random.rand(N,d),sigma)
                
                
            max_sw_temp=max_swdistance_empirical(points1,points2,N, p)
            s+= max_sw_temp
    
    return s/n_montecarlo



def expected_max_swdistance_empirical_covar_Gaussian(N, p, d=10, n_montecarlo=100):
    '''
    Compute the expection of p-max-sliced Wasserstein distance
    between two empirical measures of a multivariate Gaussian
    
    

    Parameters
    ----------
                               
    d : integer
        dimension of target distribution
    N : integer
        number of samples
    p : p>=1
        p-max-sliced Wasserstein distance
        
    n_montecarlo : TYPE
        times of montecarlo

    Returns
    -------
    positive real number
        DESCRIPTION. expection of p-max-sliced Wasserstein distance
        between two empirical measures

    '''
    
    # Covariance matrix
    mean = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    cov = np.array([
    [1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005],
    [0.5, 1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01],
    [0.3, 0.5, 1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02],
    [0.2, 0.3, 0.5, 1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03],
    [0.1, 0.2, 0.3, 0.5, 1, 0.5, 0.3, 0.2, 0.1, 0.05],
    [0.05, 0.1, 0.2, 0.3, 0.5, 1, 0.5, 0.3, 0.2, 0.1],
    [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 0.5, 0.3, 0.2],
    [0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 0.5, 0.3],
    [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 0.5],
    [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
    ])
    
    
    s=0
    
    
        
   
        
    for n in range(n_montecarlo):
       points1=np.random.multivariate_normal(mean, cov, N)
            
       points2=np.random.multivariate_normal(mean, cov, N)
            
       max_sw_temp=max_swdistance_empirical(points1,points2,N, p)
       s+= max_sw_temp 
            
   
            
    return s/n_montecarlo




def pushforward_embed_into_gaussian_rkhs(sample, theta, d_test, L, sigma_square=1, omega_square=2):
    '''
    

    Parameters
    ----------
    sample : discrete sample points in R^1
    theta : a direction vector in rkhs, 
    d_test : dimension of theta
        DESCRIPTION. The default is 5.
    sigma_square : variance of Gaussian reference measure
        DESCRIPTION. The default is 1.
    omega_square : variance of Gaussian kernel K(x,y)
        DESCRIPTION. The default is 2.

    Returns
    -------
    pushforward of points in direction theta

    '''
    
    
    #compute beta=2sigma_square/omega_square
    #beta=1
    # define some constant
    alpha=np.power(3,1/4)
    gamma=np.sqrt(3/8)
    beta=1/8
  
    
    
    position=gamma*sample
    
    # compute the pushforward points given by \sum \theta_i\psi_i(sample)
    index=np.array(range(0,d_test))
    
    # the following embedding is computed according to some theoretical results
    const_coeff=1/alpha*np.sqrt(np.power(2,index)*scipy.special.factorial(index))
    exp_sample=np.exp(-beta*position**2)
    
    coeff=np.transpose(np.reshape(np.tile(const_coeff,L),(L,d_test)))
    
    
    sample=exp_sample*np. polynomial.hermite.hermval(position,coeff*theta)

    
    
    return sample



def max_swdistance_embed_into_gaussian_rkhs(I0,I1,N, p, d_test, L=2000):
    '''
    Given two 1-dimensional sample points, I0, I1, compute p th power of
    the p-max-sliced 
    Wasserstein distance between the hilbert embeddings
    
    N: size of I0, I1 (they should have the same size)
    
    L: number of tested directions theta
    
    d_test: the dimension of theta (truncated dimension of rkhs)
    '''

    
    # eigenvalues, in fact not needed here
    #index=np.array(range(0,d_test)).astype(int)
    #Lambda=1/(2+np.sqrt(3))    
    #Lambda_coeff=np.sqrt(2*Lambda)*np.power(Lambda, index)
    
    
    #theta_mat is the matrix with each column a sample unit vector
    
    theta_mat=np.random.randn(d_test,L)
    theta_mat= theta_mat/np.sqrt(np.sum(theta_mat ** 2,axis=0))
    
    
    pushforward_points0=pushforward_embed_into_gaussian_rkhs(I0, theta_mat, d_test,L)
    pushforward_points1=pushforward_embed_into_gaussian_rkhs(I1, theta_mat, d_test,L)
    
    index_0=np.argsort(pushforward_points0,axis=0)
    index_1=np.argsort(pushforward_points1,axis=0)
        
    points_0=np.take_along_axis(pushforward_points0,index_0,axis=0)
    points_1=np.take_along_axis(pushforward_points1,index_1,axis=0)
    
    sw_piece=np.sum(np.power(abs(points_0-points_1),p),axis=0)
        
    max_sw=np.max(sw_piece)


    return np.power(max_sw/N,1/p)






def expected_max_swdistance_empirical_embed_into_gaussian_rkhs(distribution,p,N, n_montecarlo,sigma=1,d_test=30):
    '''
    

    Parameters
    ----------
    distribution: type of tested distribution, it could be uniform or Gaussian
    
    p : p-max-sliced Wasserstein distance
    N : number of sample points
    d_test : test dimension of rkhs. The default is 5.
    n_montecarlo : itereates of montecarlo
     

    Returns
    -------
    number
       expecation of
       max-sliced wasserstein distance between pushforward of the empirical
       distribution and true distribution

    '''
    
    
    s=0
    
    if distribution=='uniform':
        for n in range(n_montecarlo):
        
            points1=np.random.rand(N)
            
            points2=np.linspace(0,1,N)
        
        
            max_sw_temp=max_swdistance_embed_into_gaussian_rkhs(points1,points2, N,p, d_test)
            s+= max_sw_temp
            
    if distribution=='Gaussian':
        for n in range(n_montecarlo):
        
            points1=sigma*np.random.randn(N)
            
            points2=generate_Gaussian_vector_1d(N,sigma)
        
        
            max_sw_temp=max_swdistance_embed_into_gaussian_rkhs(points1,points2, N,p, d_test)
            s+= max_sw_temp
        
    return s/n_montecarlo




def generate_Gaussian_vector_1d(N,sigma):
    '''
    Generate an empirical measure of 1d Gaussian

    Parameters
    ----------
    N : the length of vector
    sigma^2: variance of gaussian

    Returns
    -------
    gaussian_vec : vector
        gaussian_vec is a vector consisting points in R,
        with each poins approximately 1/N mass according to 
        the Gaussian distribution

    '''
    percentile=np.linspace(1/N,1-1/N,N)
    gaussian_vec=scipy.stats.norm.ppf(percentile)
    
    return sigma*gaussian_vec















def max_swdistance_empirical_Gaussian(I0,sigma,N, p, L=2000, sample_n=1000):
    '''
    Given d-dimensional measures I0, compute the p-max-sliced 
    Wasserstein distance between it and N(0,sigma^2Id)
    
    Use the fact that normalized vectors of Gaussian variables is 
    uniformly distributed on a sphere
    
    sample_n: by default, the one-dimensional Gaussian has 1000 sample points
    '''
    
    
    d = I0.shape[1]
    
    # number of direction is related to dimension d, take 180
    # this may not be a good choice
    
    
    
    # generate a random vector on sphere, compute the wasserstein
    # distance of the pushforward measures, select the biggest 
    
    theta_mat=np.random.randn(d,L)
    theta_mat= theta_mat/np.sqrt(np.sum(theta_mat ** 2,axis=0))
    
    # generate Gaussian 
    gaussianpoints1d=generate_Gaussian_vector_1d(sample_n,sigma)
    points_1=np.transpose(np.reshape(np.tile(gaussianpoints1d,L),(L,sample_n)))
    
    
    # compute the pushforward measures
    pushforward_points0=np.dot(I0,theta_mat)
    
    
    # sort the points to obtain the optimal coupling
    index_0=np.argsort(pushforward_points0,axis=0)
    
        
    points_0=np.take_along_axis(pushforward_points0,index_0,axis=0)
    
    # notice: N should divide sample_n
    repeat_time=int(sample_n/N)
    
    points_0=np.repeat(points_0,repeat_time, axis=0)
    # p-Wasserstein distance between pushforward measures
    sw_piece=np.sum(np.power(abs(points_0-points_1),p),axis=0)
        
    max_sw=np.max(sw_piece)


    return np.power(max_sw/sample_n,1/p)


def expected_max_swdistance_gaussian(sigma, d, N, p, n_montecarlo=100):
    
    '''
    compute the expection of p-max-sliced Wasserstein distance
    between the empirical Gaussian measure and true Gaussian measure
    
    True Gaussian measure is set to be zero mean, with variance sigma^2
    '''
    s=0
    for n in range(n_montecarlo):
        points1=sigma*np.random.randn(N,d)
        
       
        max_sw_temp=max_swdistance_empirical_Gaussian(points1,sigma,N, p)
        s+= max_sw_temp 
        
    return s/n_montecarlo







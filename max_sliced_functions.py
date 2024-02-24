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



   
def max_swdistance_empirical(I0,I1,N, p, L=1000):
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


    return max_sw/N


def max_swdistance_empirical_Gaussian(I0,sigma,N, p, L=1000):
    '''
    Given d-dimensional measures I0, compute the p-max-sliced 
    Wasserstein distance between it and N(0,sigma^2Id)
    
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
    
    # generate Gaussian 
    gaussianpoints1d=generate_Gaussian_vector_1d(N,sigma)
    points_1=np.transpose(np.reshape(np.tile(gaussianpoints1d,L),(L,N)))
    
    
    # compute the pushforward measures
    pushforward_points0=np.dot(I0,theta_mat)
    
    
    # sort the points to obtain the optimal coupling
    index_0=np.argsort(pushforward_points0,axis=0)
    
        
    points_0=np.take_along_axis(pushforward_points0,index_0,axis=0)
    
    
    # p-Wasserstein distance between pushforward measures
    sw_piece=np.sum(np.power(abs(points_0-points_1),p),axis=0)
        
    max_sw=np.max(sw_piece)


    return max_sw/N



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
    between empirical measure and the target measure
    
    If sample from uniform or Gaussian, it computes the distance between 
    the empirical measure and true measure (the true measure is also 
                                            discretized, though...)
    
    If sample from Pareto, it computes the distance between two
    empirical measure, where the target distribution is the product 
    measure of indenpendent 1-dimensional Pareto distribution

    Parameters
    ----------
    distribution : string
        can be 'uniform', 'Gaussian', 'Pareta'
    sigma : positive real number
        DESCRIPTION: 'uniform': sample from U([-sigma,sigma])
                     'Gaussian': sample from N(0,sigma^2)
                     'Pareto': shape parameter of Pareto distribution
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
        between empirical measure and the target measure

    '''
    
    
    
    
    s=0
    
    if distribution=='uniform':
        for n in range(n_montecarlo):
            points1=sigma*(-1+2*np.random.rand(N,d))
        
            
            points2=np.linspace(-sigma,sigma,N)
            #if want the distance between two empirical measure
            #points2=sigma*(-1+2*np.random.rand(N,d))
            
            max_sw_temp=max_sw_temp=max_swdistance_empirical(points1,points2,N, p)
            s+= max_sw_temp
        
        
    if distribution=='Gaussian':
        
        for n in range(n_montecarlo):
            points1=sigma*np.random.randn(N,d)
            #if want the distance between two empirical measure
            #points2=sigma*np.random.randn(N,d)
            
            max_sw_temp=max_swdistance_empirical_Gaussian(points1,sigma,N, p)
            s+= max_sw_temp 
            
    if distribution=='Pareto':
        for n in range(n_montecarlo):
            
            points1=sample_from_Pareto(np.random.rand(N,d),sigma)
            points2=sample_from_Pareto(np.random.rand(N,d),sigma)
                
                
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
        DESCRIPTION. The default is 15.
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
    alpha=np.power(3,1/8)
    gamma=np.power(3/4,1/4)
  
    
    
    position=gamma*sample
    
    # compute the pushforward points given by \sum \theta_i\psi_i(sample)
    index=np.array(range(0,d_test))
    
    # the following embedding is computed according to some theoretical results
    const_coeff=alpha/np.sqrt(np.power(2,index)*scipy.special.factorial(index))
    exp_sample=np.exp(-(np.sqrt(3)-1)*position**2/4)
    
    coeff=np.transpose(np.reshape(np.tile(const_coeff,L),(L,d_test)))
    
    
    sample=exp_sample*np. polynomial.hermite.hermval(position,coeff*theta)

    
    
    return sample



def max_swdistance_embed_into_gaussian_rkhs(I0,I1,N, p, d_test, L=4000):
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


    return max_sw/N






def expected_max_swdistance_empirical_embed_into_gaussian_rkhs(distribution,p,N, n_montecarlo,sigma=1,d_test=15):
    '''
    

    Parameters
    ----------
    distribution: type of tested distribution, it could be uniform or Gaussian
    
    p : p-max-sliced Wasserstein distance
    N : number of sample points
    d_test : test dimension of rkhs. The default is 15.
    n_montecarlo : itereates of montecarlo
     

    Returns
    -------
    number
       max-sliced wasserstein distance between pushforward of the empirical
       distribution and true distribution

    '''
    
    
    s=0
    
    if distribution=='uniform':
        for n in range(n_montecarlo):
        
            points1=np.random.rand(N)
            
            points2=np.linspace(0,1,N)
        
        
            max_sw_temp=max_swdistance_embed_into_gaussian_rkhs(points1,points2, N,p, d_test)
            s+= np.power( max_sw_temp,1/p)
            
    if distribution=='Gaussian':
        for n in range(n_montecarlo):
        
            points1=3*np.random.randn(N)
            
            points2=3*generate_Gaussian_vector_1d(N,sigma)
        
        
            max_sw_temp=max_swdistance_embed_into_gaussian_rkhs(points1,points2, N,p, d_test)
            s+= np.power( max_sw_temp,1/p)
        
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




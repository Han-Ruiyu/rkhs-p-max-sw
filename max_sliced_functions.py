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
    between two empirical measures
    
    

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
        
            
            #points2=np.linspace(-sigma,sigma,N)
            #if want the distance between two empirical measure
            points2=sigma*(-1+2*np.random.rand(N,d))
            
            max_sw_temp=max_swdistance_empirical(points1,points2,N, p)
            s+= max_sw_temp
        
        
    if distribution=='Gaussian':
        
        for n in range(n_montecarlo):
            points1=sigma*np.random.randn(N,d)
            #if want the distance between two empirical measure
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



#The following functions computes the expected p-max-sliced-Wasserstein distance
#between empirical Gaussian measure and the true Gaussian measure
# in R^d, it runs slower




def inverse_density(I,N):
    ''' Given a one-dimensional density function I, 
    points in I are sorted
    return the pseudo inverse function: array N*2
    length of pseudo inverse function is N
    '''
    #return the shape of I
    points, mass=np.shape(I)
    
    #compute cdf of I
    cdf=np.zeros([points,mass])
    cdf[:,0]=I[:,0]
    cdf[:,1]=np.cumsum(I[:,1])
    
    sum_of_mass=cdf[-1,1]
    
    #rescale to ensure the total mass is 1
    cdf[:,1]=cdf[:,1]/sum_of_mass
    
    
    
    p=np.linspace(0,1,num=N,endpoint=False)
    
    pseudo_inverse=np.zeros([N,1])
    
    
    # initialize pointer
    pointer_cdf=0
    
    for j in range(N):
        while cdf[pointer_cdf,1]<p[j]:
            pointer_cdf+=1
            
        pseudo_inverse[j,0]=cdf[pointer_cdf,0]

    return pseudo_inverse


def pushforward(mu, theta):
    '''
    Given a d-dimensinal discrete measure mu, and a direction theta,
    return to its push-forward measure P_{\theta\#}mu  
    mu is in the shape array N*(d+1)
    theta is a (1,d) vector
    
    return to a 1-dimensional measure mu with sorted points and mass
    '''
    N,d=mu.shape
    support_points=mu[:,0:d-1]
    points_density=mu[:,d-1]
    
    # inner product, return the pushforward measure, 
    # N points in 1d with corresponding point mass
    support_points_pushforward=np.dot(support_points,theta)
    
    
    
    #pushforward=np.zeros([N,2])
    #pushforward[:,0]=support_points_pushforward
    #pushforward[:,1]=points_density
    #p=np.sort(pushforward,axis=0)
    # sort the density
    density={}
    for i in range(N):
        key_temp= support_points_pushforward[i]
        mass=points_density[i]
        if key_temp not in density:
            density[key_temp]=mass
        else:
            density[key_temp]+=mass
            
    myKeys = list(density.keys())
    myKeys.sort()
    k=len(myKeys)
    p=np.zeros([k,2])
    for i in range(k):
        key_i=myKeys[i]
        p[i,0]=key_i
        p[i,1]=density[key_i]
    
    return p


def inverse_density_Gaussian(sigma, num_points=100, N=500):
    ''' Given a one-dimensional Gaussian N(0, sigma^2)
    with 100 points
    return the pseudo inverse function: array N*2
    length of pseudo inverse function is N
    '''


    
    #compute cdf of I
    cdf=np.zeros([num_points,2])
    points=np.linspace(-3*sigma,3*sigma, num_points)
    cdf[:,0]=points
    cdf[:,1]=scipy.stats.norm.cdf(points)
    
    
    
    p=np.linspace(0,1,num=N,endpoint=False)
    
    pseudo_inverse=np.zeros([N,1])
    
    
    # initialize pointer
    pointer_cdf=0
    
    for j in range(N):
        while cdf[pointer_cdf,1]<p[j]:
            pointer_cdf+=1
            
        pseudo_inverse[j,0]=cdf[pointer_cdf,0]

    return pseudo_inverse


def pWasserstein_Gaussian(X, p, G, N=500):
    """Given one-dimensional density functions X, points sorted, 
   
    this function calculates the pth power of p-Wasserstein distance
    between X and one-dimensional Gaussian N(0,sigma^2)
    G is the inverse density of Gaussian generated by inverse_density_Gaussian
    
    Use the fact that the optimal coupling must be monotone
    
    N: number of intervals of integration in [0,1] of the pseudo inverse
    need to be the same as the N in inverse_density_Gaussian
    and 
    
    """
    
    I1_inverse=inverse_density(X,N)
    
    #compute the distance vector
    distance_vector=abs(G-I1_inverse)
    
    distance_p=np.power(distance_vector,p)
    
    Wp=np.sum(distance_p)
    Wp_ppower=Wp/N
    
    return Wp_ppower


def max_swdistance_Gaussian(sigma, X, p):
    '''
    Given d-dimensional measures Gaussian 
    measure, compute the p-max-sliced 
    Wasserstein distance between true distribution and sample X
    
    our distribution is the N(0,sigma^2 I_d)
    
    Use the fact that normalized vectors of Gaussian variables is 
    uniformly distributed on a sphere
    '''
    
    
    
    d = X.shape[1]-1
    
    # number of direction is related to dimension d,
    # this may not be a good choice
    L = 2000
    
    max_sw=0  
    G=inverse_density_Gaussian(sigma)
    
    for l in range(L):
        theta = np.random.randn(d)
        theta = theta / np.sqrt(np.sum(theta ** 2))
        #mu_theta=mu*np.sum(theta)
        X_temp=pushforward(X, theta)
        sw_piece=pWasserstein_Gaussian(X_temp, p, G)
        if sw_piece>=max_sw:
            max_sw=sw_piece
      
    


    return max_sw



def expected_max_swdistance_gaussian(sigma,d, N, p, n_montecarlo=100):
    '''
    Given a mu is N(0,sigma^2*I_d), 
    compute the expectation of max-sliced
    Wassersein distance between the empirical measure mu_N and mu
    Use monte-carlo to compute the expectation, no_montecarlo is 
    the time of montecarlo
    
    if it is not good then we think about something else
    
    
    '''
    # number of intervals on the probability space
    
    
    
    s=0
    
    
    mean=np.zeros(d)
    variance=sigma* np.eye(d,d)
    
    
    for n in range(n_montecarlo):
        points=np.ones([N,d+1])
        for pt in range(N):
            # generate random points
            points[pt,0:d]=np.random.multivariate_normal(mean,variance)
        
        points[:,d]=points[:,d]/N
            
        max_sw_temp=max_swdistance_Gaussian(sigma, points, p)
        s+= np.power(max_sw_temp,1/p)
    
        
    return s/n_montecarlo

max_sliced_functions.py : file of functions used to compute the max-sliced Wasserstein distance
plt_experiment_1dGaussian_rkhs: for experiments of kernel embedding, the sample space is 1d Gaussian, the kernel function for rkhs is exp(-|x-y|^2/4)
plg_experiment_Rd.py: for experiments of distribution defined on R^d

For experiments, if you want to enlarge the dimension of distributions, the corresponding number of tested direction should also be enlarged. 
This is the variable "L" in max_sliced_functions.py.

For specific functions, see the commments with each individual one.

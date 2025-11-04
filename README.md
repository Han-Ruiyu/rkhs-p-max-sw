max_sliced_functions.py : file of functions used to compute the max-sliced Wasserstein distance

plg_experiment_Rd.py: for experiments of distribution defined on R^d, in particular for figure 1-3.

plt_experiment_1dGaussian_rkhs: for experiments of kernel embedding, the sample space is 1d Gaussian, the kernel function for rkhs is exp(-|x-y|^2/4). codes for figure 4.

code-figure-5/6: as the names show, they are the codes for generating figure 5 and 6 in the paper.

For experiments, if you want to enlarge the dimension of distributions, the corresponding number of tested direction should also be enlarged. This is the variable "L" in max_sliced_functions.py.

In the RKHS embedding experiment, when the true distribution is Gaussian, the code is unstable when the variance of Gaussian is small and d_test is large.

For specific functions, see the commments within each individual one.

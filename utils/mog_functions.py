import numpy as np
import scipy
import math
import functools


#### functions for the MOG case


# parameters of the mixture 

from scipy.stats import multivariate_normal


# target density (unnormalized) with diagonal covariances
def bimodal_distribution(z, gap, sigma1, sigma2):
    # Bimodal distribution probability density function
    mean1 = [-gap/2, 0]
    cov1 = [[sigma1, 0], [0, sigma1]]
    pdf1 = multivariate_normal(mean1, cov1).pdf(z)

    mean2 = [gap/2, 0]
    
    cov2 = [[sigma2, 0], [0, sigma2]]
    pdf2 = multivariate_normal(mean2, cov2).pdf(z)

    outcome= 0.5 * pdf1 + 0.5 * pdf2
    return outcome


# target potential
#def potential(z):
#    return -np.log(bimodal_distribution(z))


# not vectorized
def gradient_of_log_bimodal_distribution(z,  gap, sigma1, sigma2):

    mean1 = np.asarray([-gap/2, 0])
    cov1 = [[sigma1, 0], [0, sigma1]]
    pdf1 = multivariate_normal(mean1, cov1).pdf(z) 
    

    mean2 = np.asarray([gap/2, 0])
    cov2 = [[sigma2, 0], [0, sigma2]]
    pdf2 = multivariate_normal(mean2, cov2).pdf(z)


    gradient= - (1/(bimodal_distribution(z, gap, sigma1, sigma2)))*(0.5*pdf1*np.matmul(np.linalg.inv(cov1),(z - mean1))\
                                                     +0.5*pdf2*np.matmul(np.linalg.inv(cov2),(z - mean2)))
    return gradient


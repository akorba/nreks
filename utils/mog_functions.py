import numpy as np
import scipy
import math
import functools


######################################
#        Exact MOG Case              #
######################################

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

    outcome = 0.5 * pdf1 + 0.5 * pdf2
    return outcome



# not vectorized
# I had to add 1e-9 to stabilize the scheme
def gradient_of_log_bimodal_distribution(z,  gap, sigma1, sigma2):

    mean1 = np.asarray([-gap/2, 0])
    cov1 = [[sigma1, 0], [0, sigma1]]
    pdf1 = multivariate_normal(mean1, cov1).pdf(z) 
    

    mean2 = np.asarray([gap/2, 0])
    cov2 = [[sigma2, 0], [0, sigma2]]
    pdf2 = multivariate_normal(mean2, cov2).pdf(z)


    gradient= - (1/(1e-9+ bimodal_distribution(z, gap, sigma1, sigma2)))*(0.5*pdf1*np.matmul(np.linalg.inv(cov1),(z - mean1))\
                                                     +0.5*pdf2*np.matmul(np.linalg.inv(cov2),(z - mean2)))
    return gradient


######################################
#          Jax MOG Case              #
######################################

import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad


# JAX-compatible multivariate normal PDF function
def jax_multivariate_normal_pdf(x, mean, cov):
    """Computes the multivariate normal PDF for JAX."""
    d = mean.shape[0]
    cov_inv = jnp.linalg.inv(cov)
    det_cov = jnp.linalg.det(cov)
    norm_factor = jnp.sqrt((2 * jnp.pi) ** d * det_cov)
    
    diff = x - mean
    exponent = -0.5 * jnp.dot(diff, jnp.dot(cov_inv, diff))
    
    return jnp.exp(exponent) / norm_factor

# JAX-compatible bimodal distribution
def bimodal_distribution_jax(z, gap, sigma1, sigma2):
    mean1 = jnp.array([-gap/2, 0])
    cov1 = jnp.array([[sigma1, 0], [0, sigma1]])

    mean2 = jnp.array([gap/2, 0])
    cov2 = jnp.array([[sigma2, 0], [0, sigma2]])

    pdf1 = jax_multivariate_normal_pdf(z, mean1, cov1)
    pdf2 = jax_multivariate_normal_pdf(z, mean2, cov2)

    return 0.5 * pdf1 + 0.5 * pdf2

def mog_potential_jax(z, gap, sigma1, sigma2):
    return -jnp.log(bimodal_distribution_jax(z, gap, sigma1, sigma2) + 1e-9)


def gradient_of_log_bimodal_distribution_jax(z, gap, sigma1, sigma2):
    """Computes the gradient of log bimodal distribution."""
    def log_density(z):
        return jnp.log(bimodal_distribution_jax(z, gap, sigma1, sigma2) + 1e-9)  # Avoid log(0)

    return grad(log_density)(z)


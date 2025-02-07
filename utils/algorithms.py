#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import math
import functools
import time
from alive_progress import alive_bar

from utils.preconditioners import *


######################################
#              MOG Case              #
######################################

from utils.mog_functions import *

def run_ULA_mog(grad_log_target, N_sim, u0, tau):
    
    d, J  = u0.shape
    us_list_ULA = np.zeros((d, J, N_sim)) 
    us_list_ULA[:,:,0] = u0 

    with alive_bar(N_sim, force_tty = True) as bar:
        for n in range(N_sim-1):    
            
            time.sleep(.001)
            us = us_list_ULA[:,:,n]
            
            vs = np.zeros_like(us)
            for i in range(us.shape[1]): # for each particle
                vs[:, i] +=  grad_log_target(us[:, i])
    
            us_list_ULA[:,:,n+1] = us_list_ULA[:,:,n] + tau*vs \
                + np.sqrt(2*tau)*np.random.normal(0,1,(d,J))
            bar()

    return us_list_ULA


# true_square_root = False corresponds to ALDI with the nonsymmetric square root of C
# while true_square_root = True corresponds to ALDI with the true square root of C
# I comment the corrective term of Nuesken
def run_ALDI_mog(grad_log_target, N_sim, u0, tau, true_square_root = True):
    
    d, J = u0.shape
    us_list_ALDI = np.zeros((d,J,N_sim))
    us_list_ALDI[:,:,0] = u0
    
    with alive_bar(N_sim, force_tty = True) as bar:
        for n in range(N_sim-1):    
            
            time.sleep(.001)
        
            us = us_list_ALDI[:,:,n]
            m_us = np.mean(us, axis=1)[:,np.newaxis]
            u_c = us - m_us 
            C = np.cov(us) * (J-1)/J 
            
            vs = np.zeros_like(us)
            for i in range(us.shape[1]):
                vs[:, i] +=  grad_log_target(us[:, i])
            
            drift = + np.dot(C,vs) #+ (d+1)*1/J*(us-m_us)  # I comment the corrective term of Nuesken
            
            # if we want to compute the nonsymmetric square root
            if not true_square_root: 
                Csqrt = 1/np.sqrt(J) * u_c
                noise = np.random.normal(0,1,(J,J))
                diff = np.sqrt(2)*Csqrt@noise
            
            else: 
                sqrtC = scipy.linalg.sqrtm(C)
                noise = np.random.normal(0,1,(2,J))
                diff = np.sqrt(2) * np.dot(sqrtC,noise)
            
        
            us_list_ALDI[:,:,n+1] = us + tau * drift  + np.sqrt(tau) * diff
            bar()
            
    return us_list_ALDI


# our scheme. with true square root of D, also without the corrective term of Nuesken
def run_ALDINR_mog(grad_log_target, N_sim, u0, tau, const):

    d, J = u0.shape    
    us_list_ALDINR = np.zeros((d,J,N_sim))
    us_list_ALDINR[:,:,0] = u0
    
    # to track the convergence
    #preconditioners = np.ones((d, d, N_sim)) # product of (D_opt+J_opt)K^{-1}
    #preconditioners[:, :, 0] = np.cov(u0)*(J-1)/J
    
    with alive_bar(N_sim, force_tty = True) as bar:
        for n in range(N_sim-1):    
            
            time.sleep(.001)
        
            us = us_list_ALDINR[:,:,n] # shape (d, J)
            m_us = np.mean(us, axis=1)[:,np.newaxis] # shape (2, 1)
            C = np.cov(us)*(J-1)/J # shape (2,2)
    
            # compute sqrt C
            sqrtC = scipy.linalg.sqrtm(C)
            
            # compute D
            D_opt_tilde, v, lambda_min = construct_D_opt_tilde(C,d)
            if lambda_min> 500:
                print("ALDINR diverging")
                
            D_opt = construct_D_opt(C,d)
    
            if np.mod(n, 500) == 0:
                print("iter")
                print(n)
                print("lambda min")
                print(lambda_min)
                #print(C)
                
            # compute psis
            psis = construct_onb(d, v)
            
            # compute sqrt D
            sqrtD = scipy.linalg.sqrtm(D_opt) # we should be able to do something cheap here
            
            # compute J opt
            J_opt = construct_J_opt(psis, v, lambda_min, const, d, sqrtC)
            
            T = J_opt + D_opt
            
            vs = np.zeros_like(us)
            for i in range(us.shape[1]):
                vs[:, i] +=  grad_log_target(us[:, i])#.reshape((2,1))
            
            drift = + np.dot(T, vs) 
            noise = np.random.normal(0,1,(2,J))
            diff = np.sqrt(2) * np.dot(sqrtD,noise) 
            
            
            us_list_ALDINR[:, :, n+1] = us + tau * drift  + np.sqrt(tau) * diff 
            
            # keep record of some stats
            #preconditioners[:, :, n] = T
            bar()
            
    return us_list_ALDINR #, preconditioners


######################################
#         Generic Case               #
######################################

from utils.grad_inference import *

import jax
import jax.numpy as jnp
#import jax.scipy as jsp
#import jax.random as jrandom
from jax import grad, vmap

"""
def compute_gradient(target_density):
    def log_density(z):
        return jnp.log(target_density(z) + 1e-9)  # Avoid log(0)
    
    return vmap(grad(log_density))  # Vectorized gradient computation
"""

def compute_gradient(target_potential):
    """ Returns a function to compute the negative gradient of the potential (∇log π(z) = -∇V(z)). """
    return jax.jit(vmap(lambda z: -grad(target_potential)(z)))  # JIT-compiled for efficiency




def run_ULA(target_potential, N_sim, u0, tau):

    d, J = u0.shape
    u0 = jax.device_put(u0)  # Move to JAX device for faster operations
    us_list_ULA = jnp.zeros((d, J, N_sim))
    us_list_ULA = us_list_ULA.at[:, :, 0].set(u0)

    #grad_log_target = jax.jit(compute_gradient(target_density))  if given as first argument a target density
    grad_log_target = jax.jit(compute_gradient(target_potential))
    
    with alive_bar(N_sim, force_tty=True) as bar:
        for n in range(N_sim - 1):
            us = us_list_ULA[:, :, n]

            # Compute gradient for all particles
            vs = grad_log_target(us.T).T  # JAX-friendly vectorized gradient computation

            # Langevin update step
            us_list_ULA = us_list_ULA.at[:, :, n + 1].set(
                us + tau * vs + jnp.sqrt(2 * tau) * jax.random.normal(jax.random.PRNGKey(n), (d, J))
            )
            bar()

    return us_list_ULA


def run_ALDI(target_potential, N_sim, u0, tau, true_square_root=True):


    d, J = u0.shape
    u0 = jax.device_put(u0)  # Move to JAX device for faster operations
    us_list_ALDI = jnp.zeros((d, J, N_sim))
    us_list_ALDI = us_list_ALDI.at[:, :, 0].set(u0)

    #grad_log_target = jax.jit(compute_gradient(target_density))  # JIT the gradient function
    grad_log_target = jax.jit(compute_gradient(target_potential))

    
    with alive_bar(N_sim, force_tty=True) as bar:
        for n in range(N_sim - 1):
            us = us_list_ALDI[:, :, n]
            m_us = jnp.mean(us, axis=1)[:, jnp.newaxis]
            u_c = us - m_us
            C = jnp.cov(us) * (J - 1) / J

            # Compute gradient for all particles at once using vmap
            vs = grad_log_target(us.T).T  # ✅ Vectorized and efficient

            drift = jnp.dot(C, vs)

            # Compute noise term
            if not true_square_root:
                Csqrt = 1 / jnp.sqrt(J) * u_c
                noise = jax.random.normal(jax.random.PRNGKey(n), (J, J))
                diff = jnp.sqrt(2) * jnp.dot(Csqrt, noise)
            else:
                sqrtC = jnp.array(scipy.linalg.sqrtm(C))
                noise = jax.random.normal(jax.random.PRNGKey(n), (d, J))
                diff = jnp.sqrt(2) * jnp.dot(sqrtC, noise)

            us_list_ALDI = us_list_ALDI.at[:, :, n + 1].set(us + tau * drift + jnp.sqrt(tau) * diff)
            bar()

    return us_list_ALDI

"""
def run_ALDINR(target_potential, N_sim, u0, tau, const, print_lambda = False):

    d, J = u0.shape
    u0 = jax.device_put(u0)  # Move data to JAX device
    us_list_ALDINR = jnp.zeros((d, J, N_sim))
    us_list_ALDINR = us_list_ALDINR.at[:, :, 0].set(u0)

    #grad_log_target = jax.jit(compute_gradient(target_density))  # JIT-compiled gradient function
    grad_log_target = jax.jit(compute_gradient(target_potential))

    with alive_bar(N_sim, force_tty=True) as bar:
        for n in range(N_sim - 1):
            #print(n)
            us = us_list_ALDINR[:, :, n]
            m_us = jnp.mean(us, axis=1)[:, jnp.newaxis]
            C = jnp.cov(us) * (J - 1) / J
            #print(C)


            # Debugging: Check for NaNs or Infs
            if jnp.isnan(C).any() or jnp.isinf(C).any():
                print("ERROR: Covariance matrix C contains NaNs or Infs!")
                print(C)
                raise ValueError("C contains NaN or Inf values!")
            
            # Debugging: Check if C is nearly singular
            cond_number = jnp.linalg.cond(C)
            print(f"Condition number of C: {cond_number}")
            if cond_number > 1e10:
                print("WARNING: Covariance matrix is nearly singular! Adding regularization.")
                C += 1e-6 * jnp.eye(C.shape[0])  # Regularization
                
            # Compute sqrtm safely
            sqrtC = jnp.array(scipy.linalg.sqrtm(C))

            # Compute diffusion matrices
            D_opt_tilde, v, lambda_min = construct_D_opt_tilde(C, d)
            if print_lambda:
                print(C)
                print(lambda_min)
            if lambda_min > 500:
                print("ALDINR diverging")

            D_opt = construct_D_opt(C, d)
            psis = construct_onb(d, v)
            sqrtD = jnp.array(scipy.linalg.sqrtm(D_opt))  # Use faster Cholesky decomposition

            # Compute J_opt
            J_opt = construct_J_opt(psis, v, lambda_min, const, d, sqrtC)
            T = J_opt + D_opt

            # Compute gradient for all particles
            vs = grad_log_target(us.T).T  # Vectorized gradient computation

            # Langevin update step
            drift = jnp.dot(T, vs)
            noise = jax.random.normal(jax.random.PRNGKey(n), (d, J))
            diff = jnp.sqrt(2) * jnp.dot(sqrtD, noise)

            us_list_ALDINR = us_list_ALDINR.at[:, :, n + 1].set(us + tau * drift + jnp.sqrt(tau) * diff)
            bar()

    return us_list_ALDINR

"""

def run_ALDINR(target_potential, N_sim, u0, tau, const, print_lambda=False):
    d, J = u0.shape
    u0 = jax.device_put(u0)  # Move data to JAX device
    us_list_ALDINR = jnp.zeros((d, J, N_sim))
    us_list_ALDINR = us_list_ALDINR.at[:, :, 0].set(u0)

    grad_log_target = jax.jit(compute_gradient(target_potential))

    with alive_bar(N_sim, force_tty=True) as bar:
        for n in range(N_sim - 1):
            us = us_list_ALDINR[:, :, n]
            m_us = jnp.mean(us, axis=1)[:, jnp.newaxis]
            C = jnp.cov(us) * (J - 1) / J

            # ✅ Detect NaNs or Infs in C
            if jnp.isnan(C).any() or jnp.isinf(C).any():
                print("ERROR: Covariance matrix C contains NaNs or Infs!")
                print(C)
                raise ValueError("C contains NaN or Inf values!")

            # ✅ Print eigenvalues for debugging
            eigvals, eigvecs = jnp.linalg.eigh(C)
            print(f"Iteration {n}: Eigenvalues of C = {eigvals}")

            # ✅ Check if covariance is nearly singular
            cond_number = jnp.linalg.cond(C)
            print(f"Iteration {n}: Condition number of C = {cond_number}")

            if cond_number > 10:
                print("WARNING: C is nearly singular, adding stronger regularization.")
                C += 1e-4 * jnp.eye(C.shape[0])  # Increase regularization
            elif cond_number > 50:
                print("EXTREME WARNING: C is almost singular! Adding even more regularization.")
                C += 1e-3 * jnp.eye(C.shape[0])  # Very strong regularization

            # ✅ Ensure C is PSD (Clip small eigenvalues)
            if jnp.min(eigvals) < 1e-5:
                print("WARNING: Small eigenvalue detected! Regularizing C more.")
                eigvals = jnp.clip(eigvals, 1e-4, None)  # Ensure minimum eigenvalue is 1e-4
                C = eigvecs @ jnp.diag(eigvals) @ eigvecs.T  # Reconstruct C

            # ✅ Use Cholesky decomposition instead of sqrtm
            try:
                sqrtC = jnp.array(scipy.linalg.cholesky(C, lower=True))
            except:
                print("Cholesky failed, switching to eigenvalue decomposition!")
                sqrtC = eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T  # Use eigvalsh

            # ✅ Check for particle collapse
            particle_std = jnp.std(us, axis=1)
            print(f"Particle std dev: {particle_std}")

            if jnp.min(particle_std) < 1e-3:
                print("Particles collapsing! Adding small jitter.")
                us += 1e-3 * jax.random.normal(jax.random.PRNGKey(n), us.shape)

            # Compute diffusion matrices
            D_opt_tilde, v, lambda_min = construct_D_opt_tilde(C, d)
            if print_lambda:
                print(C)
                print(lambda_min)
            if lambda_min > 500:
                print("ALDINR diverging")

            # ✅ Check if lambda_min is too small
            if lambda_min < 1e-4:
                print("WARNING: lambda_min is too small, adding regularization.")
                lambda_min = 1e-3

            D_opt = construct_D_opt(C, d)
            psis = construct_onb(d, v)
            sqrtD = jnp.array(scipy.linalg.sqrtm(D_opt))

            # Compute J_opt
            J_opt = construct_J_opt(psis, v, lambda_min, const, d, sqrtC)
            T = J_opt + D_opt

            # Compute gradient for all particles
            vs = grad_log_target(us.T).T  # Vectorized gradient computation

            # Langevin update step
            drift = jnp.dot(T, vs)
            noise = jax.random.normal(jax.random.PRNGKey(n), (d, J))
            diff = jnp.sqrt(2) * jnp.dot(sqrtD, noise)

            us_list_ALDINR = us_list_ALDINR.at[:, :, n + 1].set(us + tau * drift + jnp.sqrt(tau) * diff)
            bar()

    return us_list_ALDINR



######################################
#  Statistical linearization         #
######################################


import jax
import jax.numpy as jnp
import scipy.linalg
from alive_progress import alive_bar

def compute_grad_V_SL(us, G, y, Gamma_inv=None, 
                      Sigma0_inv=None, 
                      m0=None):  
    
    d, J = us.shape  # d = dimension of u, J = number of particles
    d_obs = y.shape[0]  #
    
    if Gamma_inv is None:
        Gamma_inv = jnp.eye(d_obs)  # Default: Identity matrix (d_obs, d_obs)
    if Sigma0_inv is None:
        Sigma0_inv = jnp.eye(d)  # Default: Identity matrix (d, d)
    if m0 is None:
        m0 = jnp.zeros((d, 1))  # Default: Zero vector (d, #)

        
    d, J = us.shape

    # Compute empirical means
    m_us = jnp.mean(us, axis=1, keepdims=True)  # Mean of particles (d, 1)

    # Apply G to all particles correctly (obs_dim, J)
    #G_us = jax.vmap(G, in_axes=1)(us)  # Ensure correct shape (obs_dim, J)
    #G_us = jax.vmap(lambda u: G(u).reshape(-1), in_axes=1)(us)  # Ensure (obs_dim, J)
    G_us = jax.vmap(lambda u: G(u).reshape(-1), in_axes=1)(us).reshape(-1, J)  # Explicit (obs_dim, J)

    #print("Shape of G_us:", G_us.shape)  # Debugging check

    G_mean = jnp.mean(G_us, axis=1, keepdims=True)  # Mean in observation space (obs_dim, 1)

    # Compute empirical covariance matrices
    u_c = us - m_us  # Centered particles (d, J)
    G_c = G_us - G_mean  # Centered G(u) values (obs_dim, J)

    C_t_uu = (u_c @ u_c.T) / J  # Covariance of particles (d, d)
    C_t_uy = (u_c @ G_c.T) / J  # Cross-covariance (d, obs_dim)
    C_t_yy = (G_c @ G_c.T) / J  # Covariance in G-space (obs_dim, obs_dim)

    # Compute pseudo-inverse of C_t^uu
    C_t_uu_inv = jnp.linalg.pinv(C_t_uu)

    # Compute statistical linearization gradient ∇SL G_t
    grad_SL_G = C_t_uy.T @ C_t_uu_inv  # (obs_dim, d)

    # Compute ∇V_SL using (3.7)
    residual = y - G_us  # (obs_dim, J)
    #grad_V_SL = -grad_SL_G.T @ jnp.linalg.inv(C_t_yy) @ residual - (m_us - us)
    grad_V_SL = -grad_SL_G.T @ Gamma_inv @ residual - Sigma0_inv @ (m0 - us)

    return -grad_V_SL  # Returns gradient for all particles



def run_ALDINR_SL(G, y, N_sim, u0, tau, const, Gamma_inv=None, Sigma0_inv=None, m0=None):

    d, J = u0.shape
    u0 = jax.device_put(u0)  # Move data to JAX device
    us_list_ALDINR = jnp.zeros((d, J, N_sim))
    us_list_ALDINR = us_list_ALDINR.at[:, :, 0].set(u0)

    with alive_bar(N_sim, force_tty=True) as bar:
        for n in range(N_sim - 1):
            us = us_list_ALDINR[:, :, n]  # Current particles

            # Compute statistical linearization gradient
            vs = compute_grad_V_SL(us, G, y, Gamma_inv, Sigma0_inv, m0)  # ✅ Replaces grad_log_target

            # Compute empirical covariance
            m_us = jnp.mean(us, axis=1)[:, jnp.newaxis]
            C = jnp.cov(us) * (J - 1) / J
            sqrtC = jnp.array(scipy.linalg.sqrtm(C))

            # Compute diffusion matrices
            D_opt_tilde, v, lambda_min = construct_D_opt_tilde(C, d)

            if lambda_min > 500:
                print("ALDINR diverging")

            D_opt = construct_D_opt(C, d)
            psis = construct_onb(d, v)
            sqrtD = jnp.array(scipy.linalg.sqrtm(D_opt))

            # Compute J_opt
            J_opt = construct_J_opt(psis, v, lambda_min, const, d, sqrtC)
            T = J_opt + D_opt

            # Langevin update step
            drift = jnp.dot(T, vs)
            noise = jax.random.normal(jax.random.PRNGKey(n), (2, J))
            diff = jnp.sqrt(2) * jnp.dot(sqrtD, noise)

            # Update particles
            us_list_ALDINR = us_list_ALDINR.at[:, :, n + 1].set(us + tau * drift + jnp.sqrt(tau) * diff)
            bar()

    return us_list_ALDINR


def run_ULA_SL(G, y, N_sim, u0, tau, Gamma_inv=None, Sigma0_inv=None, m0=None):

    d, J = u0.shape
    u0 = jax.device_put(u0)  # Move to JAX device for faster operations
    us_list_ULA = jnp.zeros((d, J, N_sim))
    us_list_ULA = us_list_ULA.at[:, :, 0].set(u0)

    with alive_bar(N_sim, force_tty=True) as bar:
        for n in range(N_sim - 1):
            us = us_list_ULA[:, :, n]

            # Compute statistical linearization gradient
            vs = compute_grad_V_SL(us, G, y, Gamma_inv, Sigma0_inv, m0)  # ✅ Replaces grad_log_target

            # Langevin update step
            us_list_ULA = us_list_ULA.at[:, :, n + 1].set(
                us + tau * vs + jnp.sqrt(2 * tau) * jax.random.normal(jax.random.PRNGKey(n), (d, J))
            )
            bar()

    return us_list_ULA



def run_ALDI_SL(G, y, N_sim, u0, tau, true_square_root=True):

    d, J = u0.shape
    u0 = jax.device_put(u0)  # Move to JAX device for faster operations
    us_list_ALDI_SL = jnp.zeros((d, J, N_sim))
    us_list_ALDI_SL = us_list_ALDI_SL.at[:, :, 0].set(u0)

    with alive_bar(N_sim, force_tty=True) as bar:
        for n in range(N_sim - 1):
            us = us_list_ALDI_SL[:, :, n]  # Current particles

            # Compute statistical linearization gradient of V
            vs = compute_grad_V_SL(us, G, y) #, Sigma0_inv

            # Compute empirical covariance matrix
            us = us_list_ALDI_SL[:, :, n]
            m_us = jnp.mean(us, axis=1)[:, jnp.newaxis]
            u_c = us - m_us
            C = jnp.cov(us) * (J - 1) / J
            
            vs = compute_grad_V_SL(us, G, y) 

            # Compute drift term
            #drift = C_t_uu @ vs
            drift = jnp.dot(C, vs)

            # Compute noise term
            if not true_square_root:
                Csqrt = 1 / jnp.sqrt(J) * u_c
                noise = jax.random.normal(jax.random.PRNGKey(n), (J, J))
                diff = jnp.sqrt(2) * jnp.dot(Csqrt, noise)
            else:
                sqrtC = jnp.array(scipy.linalg.sqrtm(C))
                noise = jax.random.normal(jax.random.PRNGKey(n), (d, J))
                diff = jnp.sqrt(2) * jnp.dot(sqrtC, noise)

            # Update particles
            us_list_ALDI_SL = us_list_ALDI_SL.at[:, :, n + 1].set(us + tau * drift + jnp.sqrt(tau) * diff)
            bar()

    return us_list_ALDI_SL


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import math
import functools

from utils.grad_inference import *
from utils.preconditioners import *


# independent Langevin sampler (independent particles)
#  number of iterations, initialization
def run_ULA(potential, N_sim, u0, tau):
    
    d, J  = u0.shape

    us_list_ULA = np.zeros((d, J, N_sim)) 
    us_list_ULA[:,:,0] = u0 
    
    
    for n in range(N_sim-1):    
        us = us_list_ULA[:,:,n]
        vs, _ = compute_gradients(potential, us_list_ULA[:,:,n])
        us_list_ULA[:,:,n+1] = us_list_ULA[:,:,n] - tau*vs \
            + np.sqrt(2*tau)*np.random.normal(0,1,(d,J))

    return us_list_ULA


# ALDI with nonsymmetric square root of C
def run_ALDI_with_gradient(potential, N_sim, u0, tau):
    
    d, J = u0.shape
    
    us_list_ALDI = np.zeros((d,J,N_sim))
    us_list_ALDI[:,:,0] = u0

    
    for n in range(N_sim-1):   
        us = us_list_ALDI[:,:,n]
        m_us = np.mean(us, axis=1)[:,np.newaxis]
        u_c = us-m_us 
        C = np.cov(us)*(J-1)/J 
        Csqrt = 1/np.sqrt(J)*u_c
        
        vs, _ = compute_gradients(potential, us_list_ALDI[:,:,n])
        
        drift = - np.dot(C,vs) + (d+1)*1/J*(us-m_us) 
        noise = np.random.normal(0,1,(J,J))
        diff = np.sqrt(2)*Csqrt@noise
    
        us_list_ALDI[:,:,n+1] = us+tau*drift  + np.sqrt(tau)*diff

    return us_list_ALDI

# our scheme. with true square root of D, also without the corrective term of Nuesken
def run_ALDINR(potential, N_sim, u0, tau, const):

    d, J = u0.shape    

    us_list_ALDINR = np.zeros((d,J,N_sim))
    us_list_ALDINR[:,:,0] = u0
    
    # to track the convergence
    means = np.ones((d, N_sim)) # means of particles along iterations
    means[:, 0] = np.mean(u0, axis=1)
    covariances = np.ones((d, d, N_sim))
    covariances[:, :, 0] = np.cov(u0)*(J-1)/J
    preconditioners = np.ones((d, d, N_sim)) # product of (D_opt+J_opt)K^{-1}
    preconditioners[:, :, 0] = np.cov(u0)*(J-1)/J
    
    for n in range(N_sim-1): 
        if np.mod(n,100) == 0:
            print("iter")
            print(n)
        us = us_list_ALDINR[:,:,n] # shape (d, J)
        m_us = np.mean(us, axis=1)[:,np.newaxis] # shape (2, 1)
        C = np.cov(us)*(J-1)/J # shape (2,2)

        # compute sqrt C
        sqrtC = scipy.linalg.sqrtm(C)
        
        # compute D
        D_opt_tilde, v, lambda_min = construct_D_opt_tilde(C,d)
        if lambda_min> 500:
            print("ALDINR diverging")
        print("lambda min")
        print(lambda_min)
        D_opt = construct_D_opt(C,d)
        print("trace D opt")
        print(np.trace(D_opt))
        # compute psis
        psis = construct_onb(d, v)
        
        # compute sqrt D
        sqrtD = scipy.linalg.sqrtm(D_opt)
        
        # compute J opt
        J_opt = construct_J_opt(psis, v, lambda_min, const, d, sqrtC)
        
        T = J_opt +D_opt
        vs, _ = compute_gradients(potential, us)
        
        drift = - np.dot(T,vs) 
        noise = np.random.normal(0,1,(2,J))
        diff = np.sqrt(2)*np.dot(sqrtD,noise) 
        
        us_list_ALDINR[:,:,n+1] = us+tau*drift  + np.sqrt(tau)*diff 
        
        # keep record of some stats
        means[:, n+1] = np.mean(us_list_ALDINR[:,:,n+1], axis=1)
        covariances[:,:, n+1] = np.cov(us)*(J-1)/J
        preconditioners[:, :, n+1] = T
        
    return us_list_ALDINR, means, covariances, preconditioners


"""
# EKS (ALDI) without gradient

start_time = time.time()

us_list_ALDI = run_ALDI_without_gradient(I, N_sim, u0, tau)

print(f"ALDI without gradient: {time.time()-start_time} seconds")


start_time = time.time()
us_list_ALDI = np.zeros((d,J,N_sim))
us_list_ALDI[:,:,0] = u0
total_acc = 0
tau_ALDI = tau


y_algo = np.array([[y]])
for n in range(N_sim-1):   
    us = us_list_ALDI[:,:,n]
    m_us = np.mean(us, axis=1)[:,np.newaxis]
    G_us_unprocessed = G(us)
    #if G_us_unprocessed.ndim == 1: # this is just to catch an annoying thing when G has higher dimension
    G_us = G_us_unprocessed[np.newaxis,:]
    m_G_us = np.mean(G_us, axis=1)[np.newaxis, :]
    #else:
    #    G_us = G_us_unprocessed.T
    #    m_G_us = np.mean(G_us, axis=1)[:,np.newaxis]
    #m_G_us =  np.mean(G(us)[np.newaxis,:], axis=1)[np.newaxis, :]
    u_c = us-m_us 
    g_c = G_us- m_G_us
    D = 1/J*np.einsum('ij,lj->il', u_c, g_c) 
    C = np.cov(us)*(J-1)/J 
    E = np.cov(G_us)*(J-1)/J
    Csqrt = 1/sqrt(J)*u_c

    drift = -1/(sigNoise**2+ tau_ALDI*E)*D@(G_us-y_algo) - 1/sigPrior**2*C@us + (d+1)*1/J*(us-m_us)
    noise = np.random.normal(0,1,(J,J))
    diff = sqrt(2)*Csqrt@noise

    us_list_ALDI[:,:,n+1] = us+tau_ALDI*drift  + sqrt(tau_ALDI)*diff
    
print(f"ALDI without gradient: {time.time()-start_time} seconds")
"""

"""
# EKS 3 (ALDI with square root of C)

start_time = time.time()
us_list_ALDI3 = np.zeros((d,J,N_sim))
us_list_ALDI3[:,:,0] = u0
tau_ALDI2 = tau

for n in range(N_sim-1):   
    us = us_list_ALDI3[:,:,n]
    m_us = np.mean(us, axis=1)[:,np.newaxis]  
    C = np.cov(us)*(J-1)/J 
    print(C)

    vs, H = compute_gradients(us_list_ALDI3[:,:,n])

    # compute sqrt C
    evaluesC, evectorsC = np.linalg.eig(C)
    assert (evaluesC >= 0).all()
    sqrtC = evectorsC * np.sqrt(evaluesC) @ evectorsC.T
    
    
    #drift = -1/(sigNoise**2 + tau_ALDI*E)*D@(G_us-y_algo) - 1/sigPrior**2*C@us + (d+1)*1/J*(us-m_us) # original line
    drift = - np.dot(C,vs) + (d+1)*1/J*(us-m_us) 
    noise = np.random.normal(0,1,(2,J))
    diff = sqrt(2)*np.dot(sqrtC,noise) 

    us_list_ALDI3[:,:,n+1] = us+tau_ALDI2*drift  + sqrt(tau_ALDI2)*diff
print(f"ALDI with gradient and square root of C: {time.time()-start_time} seconds")
"""

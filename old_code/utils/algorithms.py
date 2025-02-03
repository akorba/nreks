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
            for i in range(us.shape[1]):
                vs[:, i] +=  grad_log_target(us[:, i])
    
            us_list_ULA[:,:,n+1] = us_list_ULA[:,:,n] + tau*vs \
                + np.sqrt(2*tau)*np.random.normal(0,1,(d,J))
            bar()

    return us_list_ULA


# true_square_root = False corresponds to ALDI with the nonsymmetric square root of C
# while true_square_root = True corresponds to ALDI with the true square root of C
# I comment the corrective term of Nuesken
def run_ALDI_with_gradient_mog(grad_log_target, N_sim, u0, tau, true_square_root = True):
    
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


# independent Langevin sampler (independent particles)
#  number of iterations, initialization
def run_ULA(potential, N_sim, u0, tau, grad_inference = compute_gradients):
    
    potential_ = lambda u: potential(np.stack((u[0],u[1]),axis=-1)) # use this line for MOG 

    
    d, J  = u0.shape

    us_list_ULA = np.zeros((d, J, N_sim)) 
    us_list_ULA[:,:,0] = u0 
    
    
    with alive_bar(N_sim, force_tty = True) as bar:
        for n in range(N_sim-1):    
            
            time.sleep(.001)
            us = us_list_ULA[:,:,n]
            
            vs, _ = grad_inference(potential_, us_list_ULA[:,:,n])
            
            us_list_ULA[:,:,n+1] = us_list_ULA[:,:,n] - tau*vs \
                + np.sqrt(2*tau)*np.random.normal(0,1,(d,J))
            bar()

    return us_list_ULA


# true_square_root = False: ALDI with nonsymmetric square root of C
def run_ALDI_with_gradient(potential, N_sim, u0, tau, true_square_root = False, grad_inference = compute_gradients):
    
    potential_ = lambda u: potential(np.stack((u[0],u[1]),axis=-1))
    
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
            
            vs, _ = grad_inference(potential_, us_list_ALDI[:,:,n])
            
            drift = - np.dot(C,vs) + (d+1)*1/J*(us-m_us) 
            
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
def run_ALDINR(potential, N_sim, u0, tau, const, grad_inference = compute_gradients):
    
    potential_ = lambda u: potential(np.stack((u[0],u[1]),axis=-1))

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
            #reg = 1e-9 * np.eye(d)
            #C = C + reg
            
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
                print(C)
                
            # compute psis
            psis = construct_onb(d, v)
            
            # compute sqrt D
            sqrtD = scipy.linalg.sqrtm(D_opt) # we should be able to do something cheap here
            
            # compute J opt
            J_opt = construct_J_opt(psis, v, lambda_min, const, d, sqrtC)
            
            T = J_opt +D_opt
            vs, _ = grad_inference(potential_, us)
            
            drift = - np.dot(T, vs) 
            noise = np.random.normal(0,1,(2,J))
            diff = np.sqrt(2) * np.dot(sqrtD,noise) 
            
            
            us_list_ALDINR[:, :, n+1] = us + tau * drift  + np.sqrt(tau) * diff 
            
            # keep record of some stats
            #preconditioners[:, :, n] = T
            bar()
            
    return us_list_ALDINR #, preconditioners





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

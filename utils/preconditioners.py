#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import functools

def construct_D_opt_tilde(C,d):
    eigenvalues, eigenvectors = np.linalg.eig(C)
    index_min = np.argmin(eigenvalues)
    lambda_min = eigenvalues[index_min]
    v = eigenvectors[:, index_min]
    D_opt_tilde = (d/lambda_min)*np.tensordot(v, v, axes = 0) # D is symmetric 
    return D_opt_tilde, v, lambda_min

def construct_D_opt(C,d):
    D_opt_tilde, _ , lambda_min = construct_D_opt_tilde(C,d)
    D_opt = D_opt_tilde * lambda_min
    return D_opt

def construct_onb(d,v):
    psis = np.zeros((d,d))
    e_1, e_2 = np.eye(d)[:,0], np.eye(d)[:,1]

    xi = (1/np.sqrt(d))*(e_1+e_2)

    dot_product =  np.dot(v.T, xi) 

    theta =  math.acos(dot_product) 

    c, s = np.cos(theta), np.sin(theta)
    A_theta1 = np.array(((c, -s), (s, c))) 
    A_theta2 = np.array(((c, s), (-s, c)))

    if np.linalg.norm(np.dot(A_theta1,xi)-v)< 1e-8: 
        #print("original theta chosen")
        A_theta = A_theta1
    elif np.linalg.norm(np.dot(A_theta2,xi)-v)< 1e-8:
        #print("had to change theta sign")
        A_theta = A_theta2 
    else:
        print("error wrong angle computed")

    for i in range(0,d): 
        e_parallel = (np.dot(np.eye(d)[:,i],xi) - np.dot(np.eye(d)[:,i],v)*dot_product)/(1-dot_product**2) *xi \
                    +(np.dot(np.eye(d)[:,i],v) - np.dot(np.eye(d)[:,i],xi)*dot_product)/(1-dot_product**2) *v
        psi = np.dot(A_theta,e_parallel) 
        psis[:,i]= psi
    
    return psis


def construct_J_opt_tilde(psis, v, lambda_min, const, d): # works for c>1 
    J_hat = np.zeros((d,d))

    lambda_1 = 1
    lambda_2 = const**2
    lambdas = [lambda_1, lambda_2] 
    
    for j in range(d):
        for k in range(j+1,2): 
            J_hat[j,k] = -((lambdas[j]+lambdas[k])/(lambdas[j]-lambdas[k]))*np.dot(v,psis[:,j])\
                        *np.dot(v,psis[:,k])* (d/lambda_min)
            J_hat[k,j] = - J_hat[j,k]
    J_opt_tilde = functools.reduce(np.dot, [ psis, J_hat, psis.T])
    return J_opt_tilde

def construct_J_opt(psis, v, lambda_min, const, d, sqrtC):
    J_opt_tilde = construct_J_opt_tilde(psis, v, lambda_min, const, d)
    J_opt = functools.reduce(np.dot, [sqrtC, J_opt_tilde, sqrtC])
    return J_opt




# old function construct_onb with matrix V (keep this part for d>2, see mail exchange with Li Wang)

"""
def construct_onb(d,v):
    psis = np.zeros((d,d))
    e_1, e_2 = np.eye(d)[:,0], np.eye(d)[:,1]
    
    xi = (1/np.sqrt(d))*(e_1+e_2)
    
    dot_product =  np.dot(v.T, xi) 
    theta =  math.acos(dot_product) 
    
    c, s = np.cos(theta), np.sin(theta)
    A_theta1 = np.array(((c, -s), (s, c))) 
    A_theta2 = np.array(((c, s), (-s, c)))

    if np.linalg.norm(np.dot(A_theta1,xi)-v)< 1e-8: # ici c'est reverse par rapport à Li
        A_theta = A_theta1
    elif np.linalg.norm(np.dot(A_theta2,xi)-v)< 1e-8:
        A_theta = A_theta2 
    else:
        print("error wrong angle computed")
    
    difference = xi - dot_product*v 
    v_prime = difference/np.linalg.norm(difference)
    v_processed = v[np.newaxis,:].T
    v_prime_processed = v_prime[np.newaxis,:].T
    V = np.concatenate((v_processed,v_prime_processed), axis = 1) 

    T = V.dot(A_theta).dot(V.T)
    
    # test 1
    print("test of Txi = v")
    print(np.all(np.isclose(np.dot(T,xi), v)))
    
    for i in range(0,d): # J'ai pas de "eta" comme dans le code de Li
        e_parallel = np.dot(np.eye(d)[:,i],xi)*xi +np.dot(np.eye(d)[:,i],v)*v
        e_orthogonal = np.eye(d)[:,i] - e_parallel
        psi = np.dot(T,e_parallel) + e_orthogonal # Difference : Li uses A instead of T
        psis[:,i]= psi
        
    # test 2
    print("test orthonormality of psis")
    print(np.all(np.isclose(np.dot(psis,psis.T), np.eye(d))))
    
    # test 3 
    for i in range(0,d):
        np.dot(v,psis[:,i])*np.dot(v,psis[:,i])
        
    return psis
"""
#  function onb (Algorithm 3) with many tests

"""
def construct_onb(d,v):
    psis = np.zeros((d,d))
    e_1, e_2 = np.eye(d)[:,0], np.eye(d)[:,1]

    xi = (1/np.sqrt(d))*(e_1+e_2)

    dot_product =  np.dot(v.T, xi) 

    theta =  math.acos(dot_product) 

    c, s = np.cos(theta), np.sin(theta)
    A_theta1 = np.array(((c, -s), (s, c))) 
    A_theta2 = np.array(((c, s), (-s, c)))

    if np.linalg.norm(np.dot(A_theta1,xi)-v)< 1e-8: 
        #print("original theta chosen")
        A_theta = A_theta1
    elif np.linalg.norm(np.dot(A_theta2,xi)-v)< 1e-8:
        #print("had to change theta sign")
        A_theta = A_theta2 
    else:
        print("error wrong angle computed")

    #difference = xi - dot_product*v 
    #v_ortho = difference/np.linalg.norm(difference)
    #v_processed = v[np.newaxis,:].T
    #v_ortho_processed = v_ortho[np.newaxis,:].T
    #V = np.concatenate((v_processed,v_ortho_processed), axis = 1) 
    #T = V.dot(A_theta).dot(V.T)

    for i in range(0,d): 
        #e_parallel = np.dot(np.eye(d)[:,i],xi)*xi +np.dot(np.eye(d)[:,i],v)*v
        e_parallel = (np.dot(np.eye(d)[:,i],xi) - np.dot(np.eye(d)[:,i],v)*dot_product)/(1-dot_product**2) *xi \
                    +(np.dot(np.eye(d)[:,i],v) - np.dot(np.eye(d)[:,i],xi)*dot_product)/(1-dot_product**2) *v
        #e_orthogonal = np.eye(d)[:,i] - e_parallel
        psi = np.dot(A_theta,e_parallel) 
        psis[:,i]= psi
        
    # test 1
    #print("test 1 : if Txi = v")
    #print(np.all(np.isclose(np.dot(T,xi), v)))

    # test 2
    #print("test 2: orthonormality of psis")
    #print(np.all(np.isclose(np.dot(psis,psis.T), np.eye(d))))

    
    # print("test 3 : if <psi_k, Dopt_tilde psi_k> = Tr(D_opt_tilde)/d")
    #for i in range(0,d):
    #    print("test trace")
    #    a = np.dot(v,psis[:,i])*np.dot(v,psis[:,i])*(d/lambda_min)
    #    print(a)
    #    b = np.trace(D_opt_tilde)/d
    #    print(b)
    #    print(np.all(np.isclose(a,b)))
        
    # test 4 ("test 4: psis orthogonal")
    #print("test 4")
    #print(np.dot(psis[:,0],psis[:,1]))
    
    # test 5
    #print("test 4")
    #print(np.dot(psis,psis.T))
    
    return psis
"""



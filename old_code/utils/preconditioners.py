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


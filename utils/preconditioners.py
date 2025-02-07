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



def construct_onb(d,v): # note that v and v_prime are normalized
    psis = np.zeros((d,d))
    
    e_vectors = [np.eye(d)[:, i] for i in range(d)]
    xi = (1 / np.sqrt(d)) * sum(e_vectors)
    
    a = np.dot(xi, v)
    #print(a)
    v_prime = (xi - a*v)/np.linalg.norm(xi - a*v)
    b = np.dot(xi, v_prime)
    if not (-1.0 <= a <= 1.0):
        print("Warning: a is out of range for acos:", a)
    theta =  math.acos(a) 

    c, s = np.cos(theta), np.sin(theta)
    A_theta1 = np.array(((c, -s), (s, c))) 
    A_theta2 = np.array(((c, s), (-s, c)))
    
    if np.allclose(np.dot(A_theta1,np.array([a, b])), np.array([1, 0]), atol =  1e-8): 
        #print("original theta chosen")
        A_theta = A_theta1
    else:
        A_theta = A_theta2

    for i in range(0,d):
        c = np.dot(np.eye(d)[:,i],v)
        d_ = np.dot(np.eye(d)[:,i],v_prime)
        e_parallel = c*v +d_*v_prime 
        c_prime, d_prime = np.dot(A_theta, np.array([c, d_]).T)
        e_prime_parallel = c_prime*v +d_prime*v_prime
        e_orthogonal = np.eye(d)[:,i] - e_parallel # and that e_orthogonal is zero in d=2
        psi = e_prime_parallel + e_orthogonal
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


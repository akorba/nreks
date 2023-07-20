#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import functools
import matplotlib.pyplot as plt
import matplotlib.pylab as pl


def plot_results(potential, us_list, tau, name, N_burnin = 0, xmin = -2, xmax = 2, ymin = -2, ymax = 2):
    
    u0s = np.linspace(xmin,xmax,150)
    u1s = np.linspace(ymin,ymax,150)
    U0, U1 = np.meshgrid(u0s,u1s)
    U = np.stack((U0,U1))


    binsx = np.linspace(xmin,xmax,31)
    binsy = np.linspace(ymin,ymax,31)
    H, yedges, xedges = np.histogram2d(us_list[0,:,N_burnin:].flatten(),us_list[1,:,N_burnin:].flatten(), bins=[binsx,binsy])
    plt.figure()
    plt.pcolormesh(yedges, xedges, H.T, cmap=pl.cm.viridis_r); 
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    plt.contour(U0, U1, np.exp(-potential(U)), 5, alpha=0.4, colors="black")
    
    J, N_sim = us_list.shape[1], us_list.shape[2]
    plt.scatter(us_list[0, :, 0], us_list[1, :, 0], color = "purple", label = 'initial')
    plt.scatter(us_list[0, :, -1], us_list[1, :, -1], color = "red", label = 'final')
    plt.title(name+ ', J = '+str(J)+", N = "+str(N_sim) +", stepsize ="+str(tau))

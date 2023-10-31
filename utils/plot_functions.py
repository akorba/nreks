#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import functools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns

plt.rcParams["font.family"] = "Gill Sans"# Helvetica
matplotlib.rcParams.update({'font.size': 15})

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch



# plot 2D results of the algorithms
def plot_results(potential, us_list, tau, name, N_burnin = 0, xmin = -2, xmax = 2, ymin = -2, ymax = 2):
    
    plt.figure(figsize = (6, 6))
    
    potential_ = lambda u: potential(np.stack((u[0],u[1]),axis=-1))

    # plot histogram of previous positions
    binsx = np.linspace(xmin, xmax, 31)
    binsy = np.linspace(ymin, ymax, 31)
    H, yedges, xedges = np.histogram2d(us_list[0,:,N_burnin:].flatten(),us_list[1,:,N_burnin:].flatten()\
                                       , bins=[binsx,binsy])
    
    test = sns.cubehelix_palette(start = 0, rot=-.2, light = 1, as_cmap=True)
    #cmap = plt.get_cmap(light=0.5, 'YlGnBu')
    plt.pcolormesh(yedges, xedges, H.T, cmap = test); 
    
    u0s = np.linspace(xmin,xmax,150)
    u1s = np.linspace(ymin,ymax,150)
    U0, U1 = np.meshgrid(u0s,u1s)
    U = np.stack((U0,U1))
    plt.contour(U0, U1, np.exp(-potential_(U)), 5, alpha=0.4, colors="black")
    
    # plot initial and final positions
    plt.scatter(us_list[0, :, 0], us_list[1, :, 0], color = "purple", label = 'N = 0')
    plt.scatter(us_list[0, :, -1], us_list[1, :, -1], color = "red", label = 'N = 1')
    
    # parameters of the plot
    J, N_sim = us_list.shape[1], us_list.shape[2]
    plt.title(name + ', J = '+str(J)+", N = "+str(N_sim) +", $\gamma$ ="+str(tau))
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()



# plot 3D density with score
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 
    
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)



# compute at each time iteration the empirical mean of the particles
def compute_means(iterates):
    #J, N = iterates.shape[1:] # to erase
    return np.mean(iterates, axis = 1)

# compute at each time iteration the empirical covariance of the particles
def compute_covariances(iterates):
    d, J, N = iterates.shape
    covariances = np.ones((d, d, N))
    for i in range(N):
        covariances[:, :, i] = np.cov(iterates[:, :, i])*(J-1)/J
    return covariances
 
    
def plot_marginals_histogram(iterates, target_unnorm_density, name, xmin = -6, xmax = 6, ymin = -6, ymax = 6):
    
    nb_grid = 200
    u0s = np.linspace(xmin, xmax, nb_grid) # draw a grid of nb_grid points in 2d
    u1s = np.linspace(ymin, ymax, nb_grid)
    U0, U1 = np.meshgrid(u0s,u1s) # each of the Ui's is of size (200, 200) (all X coordinates of points parallel to Y axis and reverse)
    U = np.dstack((U0,U1)) # size (2, nb_grid, nb_grid)

    unnorm_dens = target_unnorm_density(U) 

    Z = np.trapz(unnorm_dens, u0s, axis=1)
    Z = np.trapz(Z, u1s)
    dens = unnorm_dens/Z 
    marg_over_x = np.trapz(dens, u0s, axis=1)
    marg_over_y = np.trapz(dens, u1s, axis=0)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(5)
    fig.set_figwidth(10)

    # marginal over x

    plt.subplot(1,2,1)
    plt.plot(u0s, marg_over_x)
    plt.hist(iterates[0,:,0:].flatten(), density=True)
    plt.plot(u0s, marg_over_x, color = 'red')
    plt.title(str(name)+' - Marginal over x')

    # marginal over y

    plt.subplot(1,2,2)
    plt.plot(u1s, marg_over_y)
    plt.hist(iterates[0,:,0:].flatten(), density=True)
    plt.plot(u1s, marg_over_y, color = 'red')
    plt.title(str(name)+' - Marginal over y')

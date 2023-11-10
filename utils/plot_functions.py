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
    
    # parameters of the plot
    J, N_sim = us_list.shape[1], us_list.shape[2]
    
    # plot initial and final positions
    plt.scatter(us_list[0, :, 0], us_list[1, :, 0], color = "purple", label = 'N = 0')
    plt.scatter(us_list[0, :, -1], us_list[1, :, -1], color = "red", label = 'N = '+str(N_sim))
    
    plt.title(name + ', J = '+str(J)+", N = "+str(N_sim) +", $\\tau$ ="+str(tau))
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.tight_layout()
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


######################################
#      Plot for statistics           #
######################################

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
 
# plots histogram of previous positions on each marginal (x, y axis in two dimensions)
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
    plt.tight_layout()
    plt.show()
    

# plot the norm of the mean of particles 
def subplot_norm_means(means_algo_all, name, J, tau, ax=None):
    
    _, N_sim, _ = means_algo_all.shape
    
    if ax is None:
        ax = plt.gca()
        
    # compute the norm (along axis 0 corresponding to d) of mean of particles 
    # at each time for each experiment - results in a vector of size (N_sim, N_exp)
    norms_mean_each_exp = np.asarray([np.linalg.norm(means_algo_all[:, i, :], axis = 0) \
                            for i in range(N_sim)])
    
    # average norm and compute std deviation over N_exp
    norm_means = np.mean(norms_mean_each_exp, axis = 1)
    std_norm_means = np.std(norms_mean_each_exp, axis = 1)
    
    subplot = ax.plot(norm_means, label = name)
    ax.fill_between(range(N_sim), norm_means - std_norm_means,\
                    norm_means + std_norm_means, alpha=0.2)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.title.set_text(r'Norm of empirical mean of particles $\Vert \bar{u}_n \Vert$  (J = '+str(J)+', $\\tau$ = '+str(tau)+')')
    return subplot

# plot the norm of the covariance over particles
def subplot_norm_covariances(covariances_algo_all, name, J, tau, ax=None):
    
    _, _, N_sim, _ = covariances_algo_all.shape
    
    if ax is None:
        ax = plt.gca()
     
    norm_covariances_each_exp = np.asarray([np.linalg.norm(covariances_algo_all[:, :, i, :], axis = (0, 1)) \
                            for i in range(N_sim)])
                                        
    norm_covariances = np.mean(norm_covariances_each_exp, axis =1)
    std_covariances = np.std(norm_covariances_each_exp, axis =1)
                                            
    subplot = ax.plot(norm_covariances, label = name)
    ax.fill_between(range(N_sim), norm_covariances - std_covariances,\
                    norm_covariances + std_covariances, alpha=0.2)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.title.set_text(r'Norm of covariance of particles $\Vert C_n^{uu}\Vert$ (J = '+str(J)+', $\\tau$ = '+str(tau)+')')
    return subplot

# plot one of the coordinates (eg x axis or y axis in 2d) of the mean of particles
def subplot_mean_coordinate(means_algo_all, name, J, tau, coordinate, ax=None):
    
    _, N_sim, _ = means_algo_all.shape
    
    if ax is None:
        ax = plt.gca()        

    mean_x_each_exp = means_algo_all[coordinate, :, :]
    
    mean_x = np.mean(mean_x_each_exp, axis = 1)
    std_mean_x = np.std(mean_x_each_exp, axis = 1)
    
    subplot = ax.plot(mean_x, label = name)
    ax.fill_between(range(N_sim), mean_x - std_mean_x,\
                    mean_x + std_mean_x, alpha=0.2)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.title.set_text('Means of particles (coordinate '+str(coordinate )+')  - (J = '+str(J)+', $\\tau$ = '+str(tau)+')')
    return subplot

# plot one of the coordinates (eg x axis or y axis in 2d) of the ERGODIC mean of particles
def subplot_ergodic_mean_coordinate(means_algo_all, name, J, tau, coordinate, ax=None):
    
    _, N_sim, _ = means_algo_all.shape
    
    if ax is None:
        ax = plt.gca()        
    
    # below it is not efficient as I am computing it for each coordinate but otherwise numpy mean flattens the array
    ergodic_mean_x_each_exp = np.asarray([np.mean(means_algo_all[:, :i, :], axis = 1)\
                                          for i in range(1, N_sim + 1)])
    ergodic_mean_x_each_exp = ergodic_mean_x_each_exp[:, coordinate, :]
    ergodic_mean_x = np.mean(ergodic_mean_x_each_exp, axis = 1)
    std_ergodic_mean_x = np.std(ergodic_mean_x_each_exp, axis = 1)
    
    subplot = ax.plot(ergodic_mean_x, label = name)
    ax.fill_between(range(N_sim), ergodic_mean_x - std_ergodic_mean_x,\
                    ergodic_mean_x + std_ergodic_mean_x, alpha=0.2)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.title.set_text('Ergodic means of particles (coordinate '+str(coordinate )+')  - (J = '+str(J)+', $\\tau$ = '+str(tau)+')')
    return subplot


def subplot_covariance_coordinate(covariances_algo_all, name, J, tau, coordinate, ax=None):
    
    i, j = coordinate
    
    _, _, N_sim, _ = covariances_algo_all.shape
    
    if ax is None:
        ax = plt.gca()        

    covariance_x_each_exp = covariances_algo_all[i, j, :, :]
    
    covariance_x = np.mean(covariance_x_each_exp, axis = 1)
    std_covariance_x = np.std(covariance_x_each_exp, axis = 1)
    
    subplot = ax.plot(covariance_x, label = name)
    ax.fill_between(range(N_sim), covariance_x - std_covariance_x,\
                    covariance_x + std_covariance_x, alpha=0.2)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.title.set_text('Covariance '+str(coordinate)+' coordinate) - (J = '+str(J)+', $\\tau$ = '+str(tau)+')')
    return subplot



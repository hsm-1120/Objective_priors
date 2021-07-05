

import pylab as plt
import numpy as np
from numba import jit, prange
from config import IM, C, path
from data import get_S_A
import stat_functions

plt.ion()
plt.show()

##
S_sq, A_sq = get_S_A(path, IM, C)

S_tot, A_tot = get_S_A(path, IM, C, quantile=0, rate=100)

init = A_tot.shape[0]
desired = 10**4
rate = desired/init*100

S, A = get_S_A(path, IM, C, quantile=0, rate=rate)

##

num_theta = 50
theta_tab = np.zeros((num_theta,2))
theta_tab[:,0] = np.linspace(0.1, A.max(), num=num_theta)
theta_tab[:,1] = np.linspace(1 /10,1/2, num=num_theta)
tmin = theta_tab.min()
tmax = theta_tab.max()
theta_grid1, theta_grid2 = np.meshgrid(theta_tab[:,0], theta_tab[:,1])

##



@jit(nopython=True)
def exponantial_prior(theta) :
    k = theta.shape[0]
    ep = np.zeros(k)
    for i in prange(k) :
        ep[i] = np.exp(-(param*theta[i]).sum()*(np.all(theta[i]>0)))*(np.all(theta[i]>0))
    return ep



# pp = np.zeros((num_theta,num_theta, A.shape[0],1))
pp = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    for j, beta in enumerate(theta_tab[:,1]) :
        # pp[i,j] = stat_functions.p_z_cond_a_theta_binary(S,A,np.array([alpha,beta]).reshape(1,2))
        pp[i,j] = stat_functions.posterior(np.array([alpha,beta]).reshape(1,2),S,A,exponantial_prior)
        # pp[i,j] = exponantial_prior(np.array([alpha,beta]).reshape(1,2))
        #pp[i,j] = p_t(S,A,np.array([alpha,beta]).reshape(1,2)).flatten()




plt.figure(1)
plt.clf()
axes = plt.axes(projection="3d")

axes.plot_surface(theta_grid1, theta_grid2, pp)

plt.title(r'posterior with exp prior, num data = {}'.format(A.shape[0]))
axes.set_xlabel(r'$\alpha$')
axes.set_ylabel(r'$\beta$')
axes.set_zlabel('p')

j_min, j_max = 0, np.max(pp)
levels = np.linspace(j_min, j_max, 15)

plt.figure(figsize=(4.5, 2.5))
plt.contourf(theta_grid1, theta_grid2, pp, cmap='viridis', levels=levels)
plt.title(r'posterior with exp prior, num data = {}'.format(A.shape[0]))
plt.axis([theta_grid1.min(), theta_grid1.max(), theta_grid2.min(), theta_grid2.max()])
plt.colorbar()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.tight_layout()
plt.show()
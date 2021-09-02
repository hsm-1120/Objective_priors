

import pylab as plt
import numpy as np
from numba import jit, prange
from config import IM, C, path
from data import get_S_A
import stat_functions

#
plt.ion()
plt.show()

##
S_sq, A_sq = get_S_A(path, IM, C)

S_tot, A_tot = get_S_A(path, IM, C, quantile=0, rate=100)

init = A_tot.shape[0]
desired = 150
rate = desired/init*100

S, A = get_S_A(path, IM, C, quantile=0, rate=rate, relative=False, shuffle=False)

##

num_theta = 50
theta_tab = np.zeros((num_theta,2))
theta_tab[:,0] = np.linspace(0.1, A.max(), num=num_theta)
theta_tab[:,1] = np.linspace(2 /10,0.8, num=num_theta)
tmin = theta_tab.min()
tmax = theta_tab.max()
theta_grid1, theta_grid2 = np.meshgrid(theta_tab[:,0], theta_tab[:,1])

##



param = np.array([1/3, 2.0])
#
@jit(nopython=True)
def exponantial_prior(theta) :
    k = theta.shape[0]
    ep = np.zeros(k)
    for i in prange(k) :
        ep[i] = np.exp(-(param*theta[i]).sum()*(np.all(theta[i]>0)))*(np.all(theta[i]>0))
    return ep



## log-likelihood

pp = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    for j, beta in enumerate(theta_tab[:,1]) :
        pp[i,j] = stat_functions.log_vrais(S,A,np.array([alpha,beta]).reshape(1,2))
pp = pp-pp.max()
ppe = np.exp(pp)


plt.figure()
plt.clf()
axes = plt.axes(projection="3d")

axes.plot_surface(theta_grid1, theta_grid2, pp.T)

plt.title(r'log-likelihood, num data = {}'.format(A.shape[0]))
axes.set_xlabel(r'$\alpha$')
axes.set_ylabel(r'$\beta$')
axes.set_zlabel('p')

j_min, j_max = 0, np.max(ppe)
levels = np.linspace(j_min, j_max, 15)

plt.figure(figsize=(4.5, 2.5))
plt.contourf(theta_grid1, theta_grid2, ppe.T, cmap='viridis', levels=levels)
plt.title(r'likelihood, num data = {}'.format(A.shape[0]))
plt.axis([theta_grid1.min(), theta_grid1.max(), theta_grid2.min(), theta_grid2.max()])
plt.colorbar()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.tight_layout()
plt.show()



## posterior with exp-prior

pp = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    for j, beta in enumerate(theta_tab[:,1]) :
        pp[i,j] = stat_functions.log_vrais(S,A,np.array([alpha,beta]).reshape(1,2)) + np.log(exponantial_prior(np.array([alpha,beta]).reshape(1,2)))
pp = pp-pp.max()
ppe = np.exp(pp)


plt.figure()
plt.clf()
axes = plt.axes(projection="3d")

axes.plot_surface(theta_grid1, theta_grid2, pp.T)

plt.title(r'log-post, prior=exp, num data = {}'.format(A.shape[0]))
axes.set_xlabel(r'$\alpha$')
axes.set_ylabel(r'$\beta$')
axes.set_zlabel('p')

j_min, j_max = 0, np.max(ppe)
levels = np.linspace(j_min, j_max, 15)

plt.figure(figsize=(4.5, 2.5))
plt.contourf(theta_grid1, theta_grid2, ppe.T, cmap='viridis', levels=levels)
plt.title(r'post, prior=exp, num data = {}'.format(A.shape[0]))
plt.axis([theta_grid1.min(), theta_grid1.max(), theta_grid2.min(), theta_grid2.max()])
plt.colorbar()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.tight_layout()
plt.show()


## Posterior with Jeffrey prior

# pp = np.zeros((num_theta,num_theta, A.shape[0],1))
from extract_saved_fisher import fisher_approx
pp = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    for j, beta in enumerate(theta_tab[:,1]) :
        # pp[i,j] = stat_functions.log_post_jeff(np.array([alpha,beta]).reshape(1,2),S,A)
        pp[i,j] = stat_functions.log_post_jeff_adapt(np.array([alpha,beta]).reshape(1,2),S,A, fisher_approx)
# pp = pp - pp.max()
ppe = np.exp(pp+25)




plt.figure()
plt.clf()
axes = plt.axes(projection="3d")

axes.plot_surface(theta_grid1, theta_grid2, pp.T)

plt.title(r'log-posterior with Jeff prior, num data={}'.format(A.shape[0]))
axes.set_xlabel(r'$\alpha$')
axes.set_ylabel(r'$\beta$')
axes.set_zlabel('p')

j_min, j_max = 0, np.max(ppe)
levels = np.linspace(j_min, j_max, 15)

plt.figure(figsize=(4.5, 3))
plt.contourf(theta_grid1, theta_grid2, ppe.T, cmap='viridis', levels=levels)
plt.title(r'post with Jeff prior, n_data={}'.format(A.shape[0]))
plt.axis([theta_grid1.min(), theta_grid1.max(), theta_grid2.min(), theta_grid2.max()])
plt.colorbar()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.tight_layout()
plt.show()

import os
import inspect

directory = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
os.chdir(directory)

import pickle
from numba import jit, prange
import numpy as np
import pylab as plt
import scipy.special as spc
from scipy import optimize

from config import path, save_fisher_arr, IM
import stat_functions
from extract_saved_fisher import fisher_approx


plt.ion()
plt.show()


## 1. Tirer des v.a. distribuées selon Jeffreys

@jit
def log_jeff(th):
    thh = np.zeros((1,2))
    thh[0] = th.flatten()
    I = fisher_approx(thh)
    return 1/2 * np.log(I[0,0,0]*I[0,1,1] - I[0,1,0]**2)


t0 = np.array([0.2,0.3])
t_fin, t_tot, acc =  stat_functions.adaptative_HM(t0, log_jeff, pi_log=True, max_iter=100000)
th = t_tot[-50000:]
n = 50000
nnn = 100000

plt.figure()
plt.plot(th[:,0],th[:,1],'o',markersize=3)

plt.figure()
plt.plot(t_tot[:,0].cumsum()/np.arange(1,nnn+1))

plt.figure()
plt.plot(np.maximum(acc,1).cumsum()/np.arange(1,nnn+1))

## 2. Calculer les coeffs d'un g-n associé


# mu = np.sum(th[:,1]**(1/2)*np.log(th[:,0])) / np.sum(np.log(th[:,0]))
mu = np.sum(th[:,1]**(1/2)*np.log(th[:,0])) / np.sum(th[:,1]**(1/2))
lamb = n/(np.sum(th[:,1]**(1/2)*(np.log(th[:,0])-mu)**2))

# pour a:
to_nul = lambda a: np.log(a) - spc.digamma(a) + np.sum(np.log(th[:,1]))/2/n - np.log(np.sum(th[:,1]**(1/2))/n)

a = optimize.newton(to_nul, 0.1)


b = n*a/np.sum(th[:,1]**(1/2))


# plt.figure()
# test = np.linspace(0.001,15,500)
# ft = np.log(test) - spc.digamma(test)
# plt.plot(test, ft)

## afficher G-N ainsi obtenu:

print(a,b,mu,lamb)


f_gn = lambda alpha, beta : 1/alpha * beta**(a/2-3/4)*np.exp(-b*beta**(1/2))*np.exp(-lamb*beta**(1/2)*(np.log(alpha)-mu)**2/2)         # * 1/2*b**a*np.sqrt(lamb)/spc.gamma(a)/np.sqrt(2*np.pi)




num_theta = 200
theta_tab = np.zeros((num_theta,2))
theta_tab[:,0] = np.linspace(10**-5, 10, num=num_theta)
theta_tab[:,1] = np.linspace(10**-1,1/2, num=num_theta)
tmin = theta_tab.min()
tmax = theta_tab.max()
theta_grid1, theta_grid2 = np.meshgrid(theta_tab[:,0], theta_tab[:,1])

pp = np.zeros((num_theta,num_theta))
# JJ = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    #th = np.concatenate((alpha*np.ones((num_theta,1)),theta_tab[:,1].reshape(num_theta,1)), axis=1)
    pp[i,:] = f_gn(alpha, theta_tab[:,1])
    # pp[i,:] = distributions.gamma_normal_density(alpha, theta_tab[:,1], mu, a, b, lamb)


plt.figure()
axes = plt.axes(projection="3d")
axes.plot_surface(theta_grid1, theta_grid2, pp.T)

plt.title('Gamma-Normal prior')
axes.set_xlabel('alpha')
axes.set_ylabel('beta')
axes.set_zlabel('p')


j_min, j_max = 0, np.max(pp)
levels = np.linspace(j_min, j_max, 15)

plt.figure(figsize=(4.5, 3))
plt.contourf(theta_grid1, theta_grid2, pp.T, cmap='viridis', levels=levels)
plt.title(r'Gamma-Normal prior')
plt.axis([theta_grid1.min(), theta_grid1.max(), theta_grid2.min(), theta_grid2.max()])
plt.colorbar()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.tight_layout()
plt.show()


### tirage du posterior:


N = 50
















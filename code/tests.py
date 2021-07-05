import os

os.chdir(r"Z:/code")

import pylab as plt
import numpy as np
import numpy.random as rd
import fisher
import numba
from numba import jit, prange
from config import IM, C, path
from data import get_S_A
import stat_functions
from utils import jrep1, jrep0

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
# a_tab, h_a = np.linspace(10**-10, 2*A.max(), num=5, retstep=True)
#
# import time
# atime = time.time()
# JJ = fisher.Fisher_Simpson_Numb(theta_tab[:,0], theta_tab[:,1], a_tab)
# btime = time.time()

##
## histogramme du posterior



def posterior_func(z, a, prior) :
    @jit
    def post(theta) :
        return stat_functions.posterior_numb(theta, z, a, prior)
    return post

# def jeff_rect_func(a_tab, h_a) :
def  jeff_rect_func2(A) :
    # @jit((numba.float32[:]))
    @jit(nopython=True, parallel=True, cache=True)
    def J(theta) :
        # theta is a kx2 array
        k = theta.shape[0]
        JJ = np.zeros(k)
        for i in prange(k) :
            if np.any(theta[i]<=0) :
                JJ[i] = 0
            else :
                alpha_tab = np.ones(1)*theta[i,0]
                beta_tab = np.ones(1)*theta[i,1]
                # JJ[i] = fisher.Jeffreys_rectangles(alpha_tab, beta_tab, a_tab, h_a)[0,0]
                # JJ[i] = fisher.Jeffreys_simpson(alpha_tab, beta_tab, a_tab)
                JJ[i] = fisher.Jeffreys_MC(alpha_tab, beta_tab, A)[0,0]
        return JJ
    return J

a_tab, h_a = np.linspace(10**-10, 2*A.max(), num=1000, retstep=True)

J = jeff_rect_func2(A)
post = posterior_func(S,A,J)

n_gen = 4000
#theta_array = stat_functions.HM_k_log_norm(np.ones(2), post, 4000, sigma_prop=np.array([[1,0],[0,0.6]]))
##
#plt.figure()

param = np.array([1/3, 2.0])

@jit(nopython=True)
def exponantial_prior(theta) :
    k = theta.shape[0]
    ep = np.zeros(k)
    for i in prange(k) :
        ep[i] = np.exp(-(param*theta[i]).sum()*(np.all(theta[i]>0)))*(np.all(theta[i]>0))
    return ep

@jit(nopython=True, parallel=True)
def HM_k_gauss(z0, pi, k, max_iter=5000, sigma_prop=10**-2*np.eye(2)) :
    d = z0.shape[0]
    z_v = jrep0(z0, k)
    for n in range(max_iter) :
        pi_zv = pi(z_v)
        z = np.zeros_like(z_v)
        for i in prange(k) :
            z[i] = z_v[i] + sigma_prop@rd.randn(d)
        pi_z = pi(z)
        alpha = pi_z / pi_zv
        rand = rd.rand(k)<alpha
        z_v += jrep1(rand,d)*(z-z_v)
    return z_v, rand, alpha, z, pi_z, pi_zv

post = posterior_func(S,A,exponantial_prior)

# theta_array = HM_k_gauss(np.ones(2), post, 30, max_iter=30, sigma_prop=np.array([[1,0],[0,0.6]]))


######

# import fisher
#
# c1 = time.time()
# J_MC = fisher.Jeffreys_MC(theta_tab[:,0], theta_tab[:,1], A)
# c2 = time.time()
#
#
# a_tab, h_a = np.linspace(10**-10, 2*A.max(), num=1000, retstep=True)
# c3 = time.time()
# J_t = fisher.Jeffreys_rectangles(theta_tab[:,0], theta_tab[:,1], a_tab, h_a)
# c4 = time.time()
#
# a_tab, h_a = np.linspace(10**-10, 2*A.max(), num=100, retstep=True)
# c5 = time.time()
# J_s = fisher.Jeffreys_simpson(theta_tab[:,0], theta_tab[:,1], a_tab)
# c6 = time.time()
#
# print(c2-c1, c4-c3, c6-c5)



####

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
# axes.plot_surface(theta_grid1, theta_grid2, np.exp(pp.mean(axis=-1)))
axes.plot_surface(theta_grid1, theta_grid2, pp)

# plt.title('Jeffreys Monte-Carlo')
axes.set_xlabel('alpha')
axes.set_ylabel('beta')
# axes.set_zlabel('J_MC')

j_min, j_max = 0, np.max(pp)
levels = np.linspace(j_min, j_max, 15)

plt.figure(figsize=(4.5, 2.5))
plt.contourf(theta_grid1, theta_grid2, pp, cmap='viridis', levels=levels)
plt.title(r'posterior with exp prior')
plt.axis([theta_grid1.min(), theta_grid1.max(), theta_grid2.min(), theta_grid2.max()])
plt.colorbar()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.tight_layout()
plt.show()


###

# import math
#
#
# @jit(nopython=True, parallel=True, cache=trie)
# def p_t(z,a,theta) :
#     l = theta.shape[0]
#     k = z.shape[0]
#     logp = np.zeros((l,k,1))
#     indnotok = np.zeros(l)
#     #for i,t in enumerate(theta) :
#     for i in prange(l) :
#         if np.any(theta[i]<=0) :
#             indnotok[i] = 1
#         else :
#             # for k,zk in enumerate(z) :
#             for k in prange(z.shape[0]) :
#                 phi = 1/2+1/2*math.erf((np.log(a[k]/theta[i,0])/theta[i,1]))
#                 if phi<1e-11 or phi>1-1e-11 :
#                     phi = (phi<1e-11)*1e-11 + (phi>1-1e-11)*(1-1e-11)
#                 logp[i,k] = z[k]*np.log(phi) + (1-z[k])*np.log((1-phi))
#     return logp



###












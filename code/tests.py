import os

os.chdir(r"Z:/code")

import pylab as plt
import numpy as np
import fisher
import numba
from numba import jit, prange
from config import IM, C, path
from data import get_S_A
import stat_functions

plt.ion()
plt.show()

##

S, A = get_S_A(path, IM, C)

##

num_theta = 3
theta_tab = np.zeros((num_theta,2))
theta_tab[:,0] = np.linspace(0.1, A.max(), num=num_theta)
theta_tab[:,1] = np.linspace(1/10,1/2, num=num_theta)
tmin = theta_tab.min()
tmax = theta_tab.max()
theta_grid1, theta_grid2 = np.meshgrid(theta_tab[:,0], theta_tab[:,1])

a_tab, h_a = np.linspace(10**-10, 2*A.max(), num=5, retstep=True)

import time
atime = time.time()
JJ = fisher.Fisher_Simpson_Numb(theta_tab[:,0], theta_tab[:,1], a_tab)
btime = time.time()

##
## histogramme du posterior



def posterior_func(z, a, prior) :
    @jit
    def post(theta) :
        return stat_functions.posterior_numb(theta, z, a, prior)
    return post

# def jeff_rect_func(a_tab, h_a) :
def  jeff_rect_func(A) :
    # @jit((numba.float32[:]))
    @jit(nopython=True, parallel=True)
    def J(theta) :
        # theta is a kx2 array
        k = theta.shape[0]
        JJ = np.zeros(k)
        for i in prange(k) :
            alpha_tab = np.ones(1)*theta[i,0]
            beta_tab = np.ones(1)*theta[i,1]
            # JJ[i] = fisher.Jeffreys_rectangles(alpha_tab, beta_tab, a_tab, h_a)[0,0]
            # JJ[i] = fisher.Jeffreys_simpson(alpha_tab, beta_tab, a_tab)
            JJ[i] = fisher.Jeffreys_MC(alpha_tab, beta_tab, A)
        return JJ
    return J

a_tab, h_a = np.linspace(10**-10, 2*A.max(), num=1000, retstep=True)

J = jeff_rect_func(A)
post = posterior_func(S,A,J)

n_gen = 4000
theta_array = stat_functions.HM_k_log_norm(np.ones(2), post, 4000)
##
plt.figure()










######

import fisher

c1 = time.time()
J_MC = fisher.Jeffreys_MC(theta_tab[:,0], theta_tab[:,1], A)
c2 = time.time()


a_tab, h_a = np.linspace(10**-10, 2*A.max(), num=1000, retstep=True)
c3 = time.time()
J_t = fisher.Jeffreys_rectangles(theta_tab[:,0], theta_tab[:,1], a_tab, h_a)
c4 = time.time()

a_tab, h_a = np.linspace(10**-10, 2*A.max(), num=100, retstep=True)
c5 = time.time()
J_s = fisher.Jeffreys_simpson(theta_tab[:,0], theta_tab[:,1], a_tab)
c6 = time.time()

print(c2-c1, c4-c3, c6-c5)








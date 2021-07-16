#import os
#os.chdir(r"Z:/code")

import numpy as np
import pylab as plt
from numba import jit
from scipy import optimize
from data import get_S_A
import reference_curves as ref
import stat_functions
from config import IM, C, path
from extract_saved_fisher import fisher_approx, jeffrey_approx

plt.ion()
plt.show()


#####
## calcul d'un maillage de Jeffrey






## fragility curve estimation via posterior estimation

S_tot, A_tot = get_S_A(path, IM, C, shuffle=False)
num_tot = A_tot.shape[0]

num_est = 50
S_ref, A_ref = S_tot[:num_tot], A_tot[:num_tot]


# tirage de nombreux MLE

def func_log_vr(z, a) :
    def log_vr(theta) :
        return stat_functions.log_vrais(z, a, theta.reshape(1,2))
    return log_vr

kmax = int(num_tot/num_est)
t0 = np.array([3,0.3])
th_MLE = np.zeros((kmax,2))
for k in range(kmax) :
    S = S_tot[k*num_est:(k+1)*num_est]
    A = A_tot[k*num_est:(k+1)*num_est]
    log_vr = func_log_vr(S,A)
    th_MLE[k] = optimize.minimize(log_vr, t0).x


# tirage d'autant de th_post

def func_log_post(z,a) :
    @jit(nopython=True)
    def log_post(theta) :
        return stat_functions.log_post_jeff_adapt(theta,z,a, Fisher=fisher_approx)
    return log_post

import time
log_post = func_log_post(S_ref, A_ref)
sig_prop = np.array([[0.25,0],[0,0.02]])
a1 = time.time()
th_post, accept = stat_functions.HM_k(t0, log_post, kmax, pi_log=True, max_iter=2000, sigma_prop=sig_prop)
a2 = time.time()

plt.figure()
plt.plot(th_MLE[:,0], th_MLE[:,1], 'x', label='MLE')
plt.plot(th_post[:,0], th_post[:,1], 'o', label='Post')
plt.title(r'Tirages de {} MLE / obs. a posteriori'.format(kmax))
plt.legend()


# tracé des quantiles de confiance comparaison avec Ref curve























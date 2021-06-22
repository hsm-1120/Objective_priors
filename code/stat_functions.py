import numpy as np
import numpy.random as rd
import math
from numba import jit, prange
from utils import simpson_numb, jrep0, jrep1
from fisher import Jeffreys_rectangles, Jeffreys_simpson
from distributions import gen_log_norm_cond, log_norm_conditionnal


## HM :

@jit(nopython=True)
def HM_gauss(z0, pi, max_iter=5000, sigma_prop=10**-2*np.eye(2)) :
    zk = z0+0
    for k in range(max_iter) :
        z = zk + sigma_prop@rd.randn(d)
        log_alpha = np.log(pi(zk))-np.log(pi(z))
        rand = np.log(rd.rand())<log_alpha
        zk += rand*(z-zk)
    return rk

@jit(nopython=True)
def HM_k(z0, pi, k, max_iter=5000, sigma_prop=10**-2) :
    d = z0.shape[0]
    z_v = jrep0(z0, k)
    for n in range(max_iter) :
        pi_zv = pi(z_v)
        z = np.zeros_like(z_v)
        for i,z_vi in enumerate(z_v) :
            z[i] = np.asarray(z_vi) + np.asarray([sigma_prop])@np.eye(d)@rd.randn(d)
        pi_z = pi(z)
        log_alpha = np.log(pi_zv)-np.log(pi_z)
        rand = np.log(rd.rand(d))<log_alpha
        #pi_z_vi = pi(z_vi)
        #pi_zi = pi(zi)
        #log_alpha = np.log(pi_z_vi)-np.log(pi_zi)
        #rand = np.log(rd.rand())<log_alpha
        z_v += jrep0(rand, k)*(z-z_v)
    return z_v

@jit(nopython=True, parallel=True)
def HM_k_log_norm(z0, pi, k, max_iter=5000, sigma_prop=10**-2) :
    d = z0.shape[0]
    z_v = jrep0(z0, k)
    for n in range(max_iter) :
        pi_zv = pi(z_v)
        z = np.zeros_like(z_v)
        for i in prange(k) :
            z[i] = gen_log_norm_cond(z_v[i], sigma_prop)
        pi_z = pi(z)
        log_alpha = np.log(pi_z)-np.log(pi_zv) + np.log(log_norm_conditionnal(z_v, z, sigma_prop)) - np.log(log_norm_conditionnal(z, z_v, sigma_prop))
        rand = np.log(rd.rand(k))<log_alpha
        #pi_z_vi = pi(z_vi)
        #pi_zi = pi(zi)
        #log_alpha = np.log(pi_z_vi)-np.log(pi_zi)
        #rand = np.log(rd.rand())<log_alpha
        z_v += jrep1(rand,d)*(z-z_v)
    return z_v



def HM_Fisher() :
    return True





## KL :

@jit(nopython=True)
def Kullback_Leibler_MC_HM(p,q, dim, iter_p=1000, iter_HM=5000) :
    p0 = np.ones(dim)
    p_simul = HM_k(p0, p, iter_p, max_iter=iter_HM)
    return np.sum(np.log(p(p_simul)/q(p_simul)))


def Kullback_Leibler_simpson(p_tab, q_tab, x_tab) :
    f = p_tab*np.log(p_tab/q_tab)
    return simpson(f, x_tab)

@jit(nopython=True, parallel=True)
def Kullback_Leibler_simpson_numb(p_tab, q_tab, x_tab) :
    f = p_tab*np.log(p_tab/q_tab)
    return simpson_numb(f, x_tab)





## distribs :

@jit(nopython=True, parallel=True, cache=True)
def p_z_cond_a_theta_binary(z,a,theta) :
    l = theta.shape[0]
    p = np.ones((l,1))
    #for i,t in enumerate(theta) :
    for i in prange(l) :
        # for k,zk in enumerate(z) :
        for k in prange(z.shape[0]) :
            phi = 1/2+1/2*math.erf((np.log(a[k]/theta[i,0])/theta[i,1])[0])
            p[i] *= z[k]*phi + (1-z[k])*(1-phi)
    return p.flatten()

def posterior(theta, z, a, prior, cond=p_z_cond_a_theta_binary) :
    p = cond(z,a,theta)*prior(theta)
    return p

@jit(nopython=True, parallel=True, cache=True)
def posterior_numb(theta, z, a, prior, cond=p_z_cond_a_theta_binary) :
    p = cond(z,a,theta)*prior(theta)
    return p
















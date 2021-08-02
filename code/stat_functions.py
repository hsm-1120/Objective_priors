import numpy as np
import numpy.random as rd
import math
from numpy.linalg import cholesky
from numba import jit, prange
from utils import simpson_numb, jrep0, jrep1
from fisher import Jeffreys_rectangles, Jeffreys_simpson, Jeffreys_simpson_numba, Fisher_Simpson, Fisher_Simpson_Numb
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
def HM_k(z0, pi, k, pi_log=False, max_iter=5000, sigma_prop=10**-2) :
    d = z0.shape[0]
    z_v = jrep0(z0, k)
    z_tot = np.zeros((max_iter,k,2))
    u = len(np.asarray(sigma_prop).shape)
    alpha_tab = np.zeros((max_iter,k))
    if u==0 :
        sig = sigma_prop*np.eye(2)
    else :
        sig = sigma_prop+0
    for n in range(max_iter) :
        pi_zv = pi(z_v)
        z = np.zeros_like(z_v)
        # for i,z_vi in enumerate(z_v) :
        for i in range(k) :
            z[i] = z_v[i] + sig@rd.randn(d)
            # z[i] = np.asarray(z_vi) + sig@rd.randn(d)
        pi_z = pi(z)
        if pi_log :
            log_alpha = pi_z - pi_zv
        else :
            log_alpha = np.log(pi_z)-np.log(pi_zv)
        rand = np.log(rd.rand(k))<log_alpha
        alpha_tab[n] = np.exp(log_alpha)
        # alpha = pi_z / pi_zv
        # rand = rd.rand(k)<alpha
        #pi_z_vi = pi(z_vi)
        #pi_zi = pi(zi)
        #log_alpha = np.log(pi_z_vi)-np.log(pi_zi)
        #rand = np.log(rd.rand())<log_alpha
        z_v += jrep1(rand, d)*(z-z_v)
        z_tot[n] = z_v + 0
    return z_v, z_tot, alpha_tab


@jit(nopython=True)
def adaptative_HM_k(z0, pi, k, pi_log=False, max_iter=5000, sigma0=0.1*np.eye(2), b=0.05) :
    d = z0.shape[0]
    z_v = jrep0(z0, k)
    z_tot = np.zeros((max_iter,k,2))
    # u = len(np.asarray(sigma0).shape)
    alpha_tab = np.zeros((max_iter,k))
    step = 40*d
    # if u==0 :
    # sig = sigma0*np.eye(d)
    # else :
    sig = sigma0+0
    sig_emp = sig+0
    for n in range(max_iter) :
        pi_zv = pi(z_v)
        z = np.zeros_like(z_v)
        # for i,z_vi in enumerate(z_v) :
        if n>=step :
            for i in prange(k) :
                z[i] = z_v[i] + (1-b)*2.38*sig_emp@rd.randn(d)/np.sqrt(d) + b* sig@rd.randn(d)/np.sqrt(d)
                # z[i] = np.asarray(z_vi) + sig@rd.randn(d)
        else :
            for i in prange(k) :
                z[i] = z_v[i] + sig@rd.randn(d)/np.sqrt(d)
        pi_z = pi(z)
        if pi_log :
            log_alpha = pi_z - pi_zv
        else :
            log_alpha = np.log(pi_z)-np.log(pi_zv)
        rand = np.log(rd.rand(k))<log_alpha
        alpha_tab[n] = np.exp(log_alpha)
        # alpha = pi_z / pi_zv
        # rand = rd.rand(k)<alpha
        #pi_z_vi = pi(z_vi)
        #pi_zi = pi(zi)
        #log_alpha = np.log(pi_z_vi)-np.log(pi_zi)
        #rand = np.log(rd.rand())<log_alpha
        z_v += jrep1(rand, d)*(z-z_v)
        z_tot[n] = z_v + 0
        # tocov = np.expand_dims(z_tot[:n+1,:,0].flatten(), axis=0)
        # be = np.expand_dims(z_tot[:n+1,:,1].flatten(), axis=0)
        tocov = np.stack((z_tot[:n+1,:,0].flatten(), z_tot[:n+1,:,1].flatten()), axis=0)
        sig_emp = cholesky(np.cov(tocov)+10**-10*np.eye(d))
        # if sig_emp.shape!=(2,2) :
        #     return sig_emp, z_tot, alpha_tab
    return z_v, z_tot, alpha_tab

@jit(nopython=True, cache=True)
def adaptative_HM(z0, pi, pi_log=False, max_iter=5000, sigma0=0.1*np.eye(2), b=0.05) :
    d = z0.shape[0]
    z_v = z0.reshape(1,2)
    z_tot = np.zeros((max_iter,2))
    # u = len(np.asarray(sigma0).shape)
    alpha_tab = np.zeros((max_iter,1))
    step = 40*d
    # if u==0 :
    # sig = sigma0*np.eye(d)
    # else :
    sig = sigma0+0
    sig_emp = sig+0
    for n in range(max_iter) :
        pi_zv = pi(z_v)
        z = np.zeros_like(z_v)
        # for i,z_vi in enumerate(z_v) :
        if n>=step :
            # for i in prange(k) :
                # z[i] = z_v[i] + (1-b)*2.38*sig_emp@rd.randn(d)/np.sqrt(d) + b* sig@rd.randn(d)/np.sqrt(d)
                # z[i] = np.asarray(z_vi) + sig@rd.randn(d)
            z[0] = z_v[0] + (1-b)*2.38*sig_emp@rd.randn(d)/np.sqrt(d) + b* sig@rd.randn(d)/np.sqrt(d)
        else :
            # for i in prange(k) :
                # z[i] = z_v[i] + sig@rd.randn(d)/np.sqrt(d)
            z[0] = z_v[0] + sig@rd.randn(d)/np.sqrt(d)
        pi_z = pi(z)
        if pi_log :
            log_alpha = pi_z - pi_zv
        else :
            log_alpha = np.log(pi_z)-np.log(pi_zv)
        # rand = np.log(rd.rand(k))<log_alpha
        rand = np.log(rd.rand())<log_alpha
        alpha_tab[n] = np.exp(log_alpha)
        # alpha = pi_z / pi_zv
        # rand = rd.rand(k)<alpha
        #pi_z_vi = pi(z_vi)
        #pi_zi = pi(zi)
        #log_alpha = np.log(pi_z_vi)-np.log(pi_zi)
        #rand = np.log(rd.rand())<log_alpha
        z_v += rand*(z-z_v)
        z_tot[n] = z_v[0] + 0
        # tocov = np.expand_dims(z_tot[:n+1,:,0].flatten(), axis=0)
        # be = np.expand_dims(z_tot[:n+1,:,1].flatten(), axis=0)
        tocov = np.stack((z_tot[:n+1,:,0].flatten(), z_tot[:n+1,:,1].flatten()), axis=0)
        sig_emp = cholesky(np.cov(tocov)+10**-10*np.eye(d))
        # if sig_emp.shape!=(2,2) :
        #     return sig_emp, z_tot, alpha_tab
    return z_v, z_tot, alpha_tab[:,0]



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

def likelihood(z, a, theta) :
    l = theta.shape[0]
    n = a.shape[0]
    p = np.zeros((l,n))
    ind0 = np.zeros(l, dtype='int')
    for i in range(l) :
        if np.any(theta[i]<=0) :
            p[i] = 0
        else :
            for k in range(n) :
                phi = 1/2+1/2*math.erf((np.log(a[k]/theta[i,0])/theta[i,1]))
                if (phi<1e-11 and z[k]==0) or (phi>1-1e-11 and z[k]==1) :
                    p[i,k] = 1
                else :
                    if (phi<1e-11 and z[k]==1) or (phi>1-1e-11 and z[k]==0) :
                        phi = (phi<1e-11)*1e-11 + (phi>1-1e-11)*(1-1e-11)
                    p[i,k] = z[k]*phi + (1-z[k])*(1-phi)
    return p.prod(axis=1)



@jit(nopython=True, parallel=True, cache=True)
def p_z_cond_a_theta_binary(z,a,theta) :
    l = theta.shape[0]
    n = a.shape[0]
    logp = np.zeros((l,n,1))
    ind0 = np.zeros(l, dtype='int')
    #for i,t in enumerate(theta) :
    for i in range(l) :
        if np.any(theta[i]<=0) :
            ind0[i] = 1
        else :
            # for k,zk in enumerate(z) :
            for k in prange(n) :
                phi = 1/2+1/2*math.erf((np.log(a[k]/theta[i,0])/theta[i,1]))
                if phi<1e-11 or phi>1-1e-11 :
                    phi = (phi<1e-11)*1e-11 + (phi>1-1e-11)*(1-1e-11)
                logp[i,k] = z[k]*np.log(phi) + (1-z[k])*np.log((1-phi))
    p = np.exp(logp.sum(axis=1)).flatten()
    # print((p==0).sum())
    # p[ind0] = 0
    p = p*(1-ind0)
    # print(ind0)
    # print((p==0).sum())
    return p

def posterior(theta, z, a, prior, cond=p_z_cond_a_theta_binary) :
    p = cond(z,a,theta)*prior(theta)
    return p

@jit(nopython=True, parallel=True, cache=True)
def posterior_numb(theta, z, a, prior, cond=p_z_cond_a_theta_binary) :
    p = cond(z,a,theta)*prior(theta)
    return p





@jit(nopython=True, parallel=True, cache=True)
def log_vrais(z,a,theta) :
    l = theta.shape[0]
    n = a.shape[0]
    logp = np.zeros((l,n,1))
    ind0 = np.zeros(l, dtype=np.int64)
    #for i,t in enumerate(theta) :
    for i in range(l) :
        if np.any(theta[i]<=0) :
            ind0[i] = 1
        else :
            # for k,zk in enumerate(z) :
            for k in prange(n) :
                phi = 1/2+1/2*math.erf((np.log(a[k]/theta[i,0])/theta[i,1]))
                if phi<1e-11 or phi>1-1e-11 :
                    phi = (phi<1e-11)*1e-11 + (phi>1-1e-11)*(1-1e-11)
                logp[i,k] = z[k]*np.log(phi) + (1-z[k])*np.log((1-phi))
    # logp = logp - logp.max()
    lp = (logp.sum(axis=1)).flatten()
    # m = lp.max()
    # lp = lp - lp.max()
    lp = lp - ind0*1e11
    return lp


@jit(nopython=True, parallel=True, cache=True)
def log_post_jeff(theta, z, a) :
    l = theta.shape[0]
    # if Jeffrey :
    log_J = np.zeros(l)
    # for i, t in enumerate(theta) :
    for i in prange(l) :
        t = theta[i]
        alpha = t[0]
        beta = t[1]
        if alpha<=0 or beta<=0 :
            log_J[i] = -1e-11
        else :
            a_tab = np.exp(np.linspace(np.log(alpha)-4*beta, np.log(alpha)+4*beta, 40))
            I = Fisher_Simpson_Numb(np.array([alpha]), np.array([beta]), a_tab)
            log_J[i] = 1/2 * np.log(I[0,0,0,0]*I[0,0,1,1] - I[0,0,1,0]**2) #/a.shape[0]
    # else :
    #     log_J = prior(theta)/a.shape[0]
    vr = log_vrais(z,a,theta) #/a.shape[0]
    return vr + log_J


def log_post_jeff_notnumb(theta, z, a) :
    l = theta.shape[0]
    # if Jeffrey :
    log_J = np.zeros(l)
    # for i, t in enumerate(theta) :
    for i in range(l) :
        t = theta[i]
        alpha = t[0]
        beta = t[1]
        a_tab = np.exp(np.linspace(np.log(alpha)-4*beta, np.log(alpha)+4*beta, 40))
        I = Fisher_Simpson(np.array([alpha]), np.array([beta]), a_tab)
        log_J[i] = 1/2 * np.log(I[0,0,0,0]*I[0,0,1,1] - I[0,0,1,0]**2)#/a.shape[0]
    # else :
    #     log_J = prior(theta)/a.shape[0]
    vr = log_vrais(z,a,theta) #/a.shape[0]
    return vr + log_J



@jit(nopython=True, parallel=True, cache=True)
def log_post_jeff_adapt(theta, z, a, Fisher) :
    I = Fisher(theta)
    log_J = 1/2 * np.log(I[:,0,0]*I[:,1,1] - I[:,1,0]**2)
    vr = log_vrais(z,a,theta)
    return vr + log_J





















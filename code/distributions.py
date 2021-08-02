from numba import jit
import numpy as np
import numpy.random as rd


# probability densities / generating functions in numba for HM algs
# densities calculated up to a constant

@jit(nopython=True)
def gauss_density(x, mu, sigma) :
    return np.exp(-(x-mu)**2/2/sigma**2)

def mult_gauss_density() :
    return True

@jit(nopython=True)
def chi2_density(x, k) :
    return x**(k/2-1) * np.exp(-x/2)


@jit(nopython=True, parallel=True, cache=True)
def log_norm_density(x, mu, sigma) : #compatibility with multivariate while there are mutual independance
    return 1/x * np.exp(-np.sum((np.log(x)-mu)**2, axis=0)/2/sigma**2)


@jit(nopython=True, cache=True)
def gamma_normal_density(mu, tau, m, a, b, lamb) :

    return tau**(a-1/2) * np.exp(-b*tau) * np.exp(-lamb*tau*(mu-m)**2/2)

@jit(nopython=True, cache=True)
def log_gamma_normal_pdf(mu, tau, m, a, b, lamb) :
    l = mu.shape[0]
    tab = np.zeros(l)
    for i in range(l):
        if np.isnan(mu[i]) or tau[i]<=0 :
            tab[i] = -10**5
        else :
            tab[i] = np.log(gamma_normal_density(mu[i], tau[i], m, a, b, lamb))
    return tab

#conditionnal usage for HM

@jit(nopython=True)
def chi2_conditionnal(z1, z2) :
    k = z1.shape[1]
    lamb = z2/k
    return chi2_density(z1/lamb, k)/lamb

@jit(nopython=True)
def gauss_conditionnal(z1, z2, sigma_prop) :
    return gauss_density(z1, 2, sigma_prop)

@jit(nopython=True, parallel=True, cache=True)
def log_norm_conditionnal(z1, z2, sigma_prop) :
    d = z2.shape[1]
    return log_norm_density(z1, np.log(z2), sigma_prop).sum(axis=1)



# generators
@jit(nopython=True)
def gen_gauss() :
    return True

@jit(nopython=True)
def gen_chi2(z1, k) :
    return True

def gen_cond_chi2(z1, z2) :
    return True

@jit(nopython=True, parallel=True, cache=True)
def gen_log_norm(mu, sigma) :
    d = mu.shape[0]
    return np.exp(mu + sigma * rd.randn(d))

@jit(nopython=True, parallel=True, cache=True)
def gen_log_norm_cond(z2, sigma_prop) :
    return gen_log_norm(np.log(z2), sigma_prop)







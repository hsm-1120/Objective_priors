import os

os.chdir(r"Z:/code")

import pylab as plt
import numpy as np
import numpy.random as rd
import math
import fisher
import numba
from numba import jit, prange
from config import IM, C, path
from data import get_S_A
import stat_functions
from utils import jrep1, jrep0
from scipy import optimize
from scipy.special import erf
# from distributions import gamma_normal_density
import distributions


plt.ion()
plt.show()

##
S_sq, A_sq = get_S_A(path, IM, C)

S_tot, A_tot = get_S_A(path, IM, C, quantile=0, rate=100)

init = A_tot.shape[0]
desired = 50
rate = desired/init*100

S, A = get_S_A(path, IM, C, quantile=0, rate=rate, shuffle = False, relative=False)

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
axes.plot_surface(theta_grid1, theta_grid2, pp.T)

# plt.title('Jeffreys Monte-Carlo')
axes.set_xlabel('alpha')
axes.set_ylabel('beta')
# axes.set_zlabel('J_MC')

j_min, j_max = 0, np.max(pp)
levels = np.linspace(j_min, j_max, 15)

plt.figure(figsize=(4.5, 2.5))
plt.contourf(theta_grid1, theta_grid2, pp.T, cmap='viridis', levels=levels)
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

# calcul du max de vraissemblance


def opp_log_vraiss(theta) :
    a = A+0
    z = S+0
    # l = theta.shape[0]
    n = a.shape[0]
    logp = np.zeros((n,1))
    # ind0 = np.zeros(l, dtype='int')
    #for i,t in enumerate(theta) :
    if np.any(theta<=0) :
        return np.inf
    else:
        # for k,zk in enumerate(z) :
        for k in prange(n) :
            phi = 1/2+1/2*math.erf((np.log(a[k]/theta[0])/theta[1]))
            if phi<1e-11 or phi>1-1e-11 :
                phi = (phi<1e-11)*1e-11 + (phi>1-1e-11)*(1-1e-11)
            logp[k] = z[k]*np.log(phi) + (1-z[k])*np.log((1-phi))
    # p = np.exp(logp.sum(axis=1)/n).flatten()
    # print((p==0).sum())
    # p[ind0] = 0
    # p = p*(1-ind0)
    # print(ind0)
    # print((p==0).sum())
    return -logp.mean(axis=0).flatten()



theta_ML = optimize.minimize(opp_log_vraiss, np.array([3,0.3]), tol=10**-5)



pp = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    for j, beta in enumerate(theta_tab[:,1]) :
        # pp[i,j] = stat_functions.p_z_cond_a_theta_binary(S,A,np.array([alpha,beta]).reshape(1,2))
        # pp[i,j] = stat_functions.log_vrais(S,A,np.array([alpha,beta]).reshape(1,2))
        pp[i,j] = opp_log_vraiss(np.array([alpha, beta]))
        # pp[i,j] = stat_functions.posterior(np.array([alpha,beta]).reshape(1,2),S,A,exponantial_prior)
        # pp[i,j] = exponantial_prior(np.array([alpha,beta]).reshape(1,2))
        #pp[i,j] = p_t(S,A,np.array([alpha,beta]).reshape(1,2)).flatten()
# pp = pp-pp.max()


plt.figure()
axes = plt.axes(projection="3d")

axes.plot_surface(theta_grid1, theta_grid2, -pp.T)

plt.title(r'log vraiss, num data = {}'.format(A.shape[0]))
axes.set_xlabel(r'$\alpha$')
axes.set_ylabel(r'$\beta$')
axes.set_zlabel('p')


pp = pp-pp.min()


ppe = np.exp(-pp*A.shape[0])


plt.figure()
plt.clf()
axes = plt.axes(projection="3d")

axes.plot_surface(theta_grid1, theta_grid2, ppe.T)

plt.title(r'vraiss, num data = {}'.format(A.shape[0]))
axes.set_xlabel(r'$\alpha$')
axes.set_ylabel(r'$\beta$')
axes.set_zlabel('p')


##

JJ = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    for j, beta in enumerate(theta_tab[:,1]) :
        a_tab = np.exp(np.linspace(np.log(alpha)-4*beta, np.log(alpha)+4*beta, 40))
        # pp[i,j] = stat_functions.p_z_cond_a_theta_binary(S,A,np.array([alpha,beta]).reshape(1,2))
        # pp[i,j] = stat_functions.log_vrais(S,A,np.array([alpha,beta]).reshape(1,2))
        # pp[i,j] = opp_log_vraiss(np.array([alpha, beta]))
        JJ[i,j] = np.log(fisher.Jeffreys_simpson(np.array([alpha]).reshape(1,1), np.array([alpha]).reshape(1,1), a_tab))
        # pp[i,j] = stat_functions.posterior(np.array([alpha,beta]).reshape(1,2),S,A,exponantial_prior)
        # pp[i,j] = exponantial_prior(np.array([alpha,beta]).reshape(1,2))
        #pp[i,j] = p_t(S,A,np.array([alpha,beta]).reshape(1,2)).flatten()
# pp = pp-pp.max()

ppj = -pp*A.shape[0] + JJ

plt.figure()
axes = plt.axes(projection="3d")

axes.plot_surface(theta_grid1, theta_grid2, ppj.T)

plt.title(r'log post (prior=J), num data = {}'.format(A.shape[0]))
axes.set_xlabel(r'$\alpha$')
axes.set_ylabel(r'$\beta$')
axes.set_zlabel('p')


plt.figure()
axes = plt.axes(projection="3d")

axes.plot_surface(theta_grid1, theta_grid2, np.exp(ppj-ppj.max()).T)

plt.title(r'post (prior=J), num data = {}'.format(A.shape[0]))
axes.set_xlabel(r'$\alpha$')
axes.set_ylabel(r'$\beta$')
axes.set_zlabel('p')

##
#MLE with vraiss (not log vraiss)

n = A.shape[0]

def opp_vraiss(theta) :
    a = A+0
    z = S+0
    # l = theta.shape[0]
    n = a.shape[0]
    logp = np.zeros((n,1))
    # ind0 = np.zeros(l, dtype='int')
    #for i,t in enumerate(theta) :
    if np.any(theta<=0) :
        return -np.inf
    else:
        # for k,zk in enumerate(z) :
        t = 0
        for k in prange(n) :
            phi = 1/2+1/2*math.erf((np.log(a[k]/theta[0])/theta[1]))
            if (phi<1e-11 and z[k]==0) or (phi>1-1e-11 and z[k]==1) :
                logp[k] = 0
                t+=1
            else :
                if (phi<1e-11 and z[k]==1) or (phi>1-1e-11 and z[k]==0) :
                    phi = (phi<1e-11)*1e-11 + (phi>1-1e-11)*(1-1e-11)
                logp[k] = z[k]*np.log(phi) + (1-z[k])*np.log((1-phi))
    # p = np.exp(logp.sum(axis=1)/n).flatten()
    # print((p==0).sum())
    # p[ind0] = 0
    # p = p*(1-ind0)
    # print(ind0)
    # print((p==0).sum())
    # logp = logp - logp.max()
    # print(t)
    return -np.exp(logp.mean(axis=0).flatten())#+62)




def jac_vrais(theta) :
    alph = theta[0]
    bet = theta[1]
    gam = np.log(A/alph)/bet
    phi = 1/2+1/2*erf(gam)
    phi = (phi<1e-11)*1e-11 + (phi>1-1e-11)*(1-1e-11) + ((phi<1-1e-11)*(phi>1e-11))*phi
    partial_alpha_logp = -1/(alph*bet) * S * np.exp(-gam**2)/np.sqrt(np.pi)/phi + 1/(alph*bet) * (1-S) * np.exp(-gam**2)/np.sqrt(np.pi)/(1-phi)
    partial_bet_logp = -gam/bet * S * np.exp(-gam**2)/np.sqrt(np.pi)/phi + gam/bet * (1-S) * np.exp(-gam**2)/np.sqrt(np.pi)/(1-phi)
    j1 = partial_alpha_logp.sum()*opp_vraiss(theta)
    j2 = partial_bet_logp.sum()*opp_vraiss(theta)
    return np.array([j1,j2])



theta_ML2 = optimize.minimize(opp_vraiss, np.array([8,0.27]))


ee = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    for j, beta in enumerate(theta_tab[:,1]) :
        # pp[i,j] = stat_functions.p_z_cond_a_theta_binary(S,A,np.array([alpha,beta]).reshape(1,2))
        # pp[i,j] = stat_functions.log_vrais(S,A,np.array([alpha,beta]).reshape(1,2))
        # pp[i,j] = opp_log_vraiss(np.array([alpha, beta]))
        ee[i,j] = opp_vraiss(np.array([alpha, beta]))
        # pp[i,j] = stat_functions.posterior(np.array([alpha,beta]).reshape(1,2),S,A,exponantial_prior)
        # pp[i,j] = exponantial_prior(np.array([alpha,beta]).reshape(1,2))
        #pp[i,j] = p_t(S,A,np.array([alpha,beta]).reshape(1,2)).flatten()
# pp = pp-pp.max()


plt.figure()
axes = plt.axes(projection="3d")

axes.plot_surface(theta_grid1, theta_grid2, -ee.T)

plt.title(r'vraiss, num data = {}'.format(A.shape[0]))
axes.set_xlabel(r'$\alpha$')
axes.set_ylabel(r'$\beta$')
axes.set_zlabel('p')


##

ll = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    for j, beta in enumerate(theta_tab[:,1]) :
        # pp[i,j] = stat_functions.p_z_cond_a_theta_binary(S,A,np.array([alpha,beta]).reshape(1,2))
        # pp[i,j] = stat_functions.log_vrais(S,A,np.array([alpha,beta]).reshape(1,2))
        # pp[i,j] = opp_log_vraiss(np.array([alpha, beta]))
        ll[i,j] = stat_functions.likelyhood(S,A,np.array([[alpha, beta]]))
        # pp[i,j] = stat_functions.posterior(np.array([alpha,beta]).reshape(1,2),S,A,exponantial_prior)
        # pp[i,j] = exponantial_prior(np.array([alpha,beta]).reshape(1,2))
        #pp[i,j] = p_t(S,A,np.array([alpha,beta]).reshape(1,2)).flatten()
# pp = pp-pp.max()


plt.figure()
axes = plt.axes(projection="3d")

axes.plot_surface(theta_grid1, theta_grid2, ll.T)

plt.title(r'vraiss, num data = {}'.format(A.shape[0]))
axes.set_xlabel(r'$\alpha$')
axes.set_ylabel(r'$\beta$')
axes.set_zlabel('p')

def liky(theta):
    # return stat_functions.likelyhood(S,A,theta.reshape(1,2))
    return likelyhood(S,A, theta.reshape(1,2))

th_last = optimize.minimize(liky, np.array([3,0.4]), tol=10**-50)

##

# log_post_tracés et max

def opp_log_post(theta) :
    I = fisher.Fisher_Simpson(theta[0].reshape(1,1), theta[1].reshape(1,1), a_tab)
    ovr = opp_log_vraiss(theta)
    J = 1/2 * np.log(I[0,0,0,0]*I[0,0,1,1] - I[0,0,1,0]**2)/A.shape[0]
    return ovr - J

th_post_ML = optimize.minimize(opp_log_post, np.array([3,0.3]))



##

# moyennes du post

post = posterior_func(S,A,exponantial_prior)

post_the = HM_k_gauss(np.ones(2), post, 30, max_iter=30, sigma_prop=np.array([[1,0],[0,0.6]]))


plt.figure()
plt.plot(post_the[:,0], post_the[:,1])
plt.title(r'$\theta\sim p(\theta|S,A)$, prior=exp, num data={}'.format(A.shpae[0]))


##

pp = np.zeros((num_theta,num_theta))
pp2 = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    for j, beta in enumerate(theta_tab[:,1]) :
        # pp[i,j] = stat_functions.p_z_cond_a_theta_binary(S,A,np.array([alpha,beta]).reshape(1,2))
        # pp[i,j] = stat_functions.log_vrais(S,A,np.array([alpha,beta]).reshape(1,2))
        pp[i,j] = stat_functions.log_vrais(S,A, np.array([[alpha, beta]]))
        # pp[i,j] = stat_functions.posterior(np.array([alpha,beta]).reshape(1,2),S,A,exponantial_prior)
        # pp[i,j] = exponantial_prior(np.array([alpha,beta]).reshape(1,2))
        #pp[i,j] = p_t(S,A,np.array([alpha,beta]).reshape(1,2)).flatten()
# pp = pp-pp.max()

for i,alpha in enumerate(theta_tab[:,0]) :
    for j, beta in enumerate(theta_tab[:,1]) :
        # pp[i,j] = stat_functions.p_z_cond_a_theta_binary(S,A,np.array([alpha,beta]).reshape(1,2))
        # pp[i,j] = stat_functions.log_vrais(S,A,np.array([alpha,beta]).reshape(1,2))
        pp2[i,j] = stat_functions.log_post_jeff(np.array([[alpha, beta]]),S,A)
        # pp[i,j] = stat_functions.posterior(np.array([alpha,beta]).reshape(1,2),S,A,exponantial_prior)
        # pp[i,j] = exponantial_prior(np.array([alpha,beta]).reshape(1,2))
        #pp[i,j] = p_t(S,A,np.array([alpha,beta]).reshape(1,2)).flatten()
# pp = pp-pp.max()




plt.figure()
axes = plt.axes(projection="3d")

axes.plot_surface(theta_grid1, theta_grid2, pp.T)

plt.title(r'log vraiss, num data = {}'.format(A.shape[0]))
axes.set_xlabel(r'$\alpha$')
axes.set_ylabel(r'$\beta$')
axes.set_zlabel('p')




plt.figure()
axes = plt.axes(projection="3d")

axes.plot_surface(theta_grid1, theta_grid2, pp2.T)

plt.title(r'log post, num data = {}'.format(A.shape[0]))
axes.set_xlabel(r'$\alpha$')
axes.set_ylabel(r'$\beta$')
axes.set_zlabel('p')



## Plot KL(posterior|Jeffreys) as a function of the number of observations

# dim_k = 1000
# range_k = np.arange(1000)*100 # len(S) = 10000
#
# #jit
# def iterate_KL(range_k, dim_k) :
#   KL = np.zeros(dim_k)
#   for i,k in enumerate(range_k) :
#     z = S[:k,0]
#     a = A[:k,0]
#     J = Jeffreys_func(A[:,0])
#     theta_simul = HM_k(np.ones(2), J, 1000)
#     post_const = np.mean(p_z_cond_a_theta_binary(z,a,theta_simul))
#     KL[i] = Kullback_Leibler(posterior_func(z, a, post_const, J)/post_const, J)
##

S_tot, A_tot = get_S_A(path, IM, C, quantile=0, rate=100)
init = A_tot.shape[0]
desired = 50
rate = desired/init*100

S, A = get_S_A(path, IM, C, quantile=0, rate=rate, shuffle = False, relative=False)


t0 = jrep0(np.array([3,0.3]),2000)
a1 = time.time()
h = stat_functions.log_post_jeff(t0, S,A)
a2 = time.time()
print(h, a2-a1)



####

m = 0
a = 0.01
b = 0.01
lamb = 0.01

pp = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    for j, beta in enumerate(theta_tab[:,1]) :
        # pp[i,j] = stat_functions.p_z_cond_a_theta_binary(S,A,np.array([alpha,beta]).reshape(1,2))
        # pp[i,j] = stat_functions.log_vrais(S,A,np.array([alpha,beta]).reshape(1,2))
        # pp[i,j] = stat_functions.log_vrais(S,A, np.array([[alpha, beta]])) + np.log(distributions.gamma_normal_density(np.log(alpha), beta, m, a, b, lamb))
        pp[i,j] = np.log(distributions.gamma_normal_density(np.log(alpha), beta, m, a, b, lamb))
        # pp[i,j] = stat_functions.posterior(np.array([alpha,beta]).reshape(1,2),S,A,exponantial_prior)
        # pp[i,j] = exponantial_prior(np.array([alpha,beta]).reshape(1,2))
        #pp[i,j] = p_t(S,A,np.array([alpha,beta]).reshape(1,2)).flatten()
# pp = pp-pp.max()


plt.figure()
axes = plt.axes(projection="3d")

axes.plot_surface(theta_grid1, theta_grid2, np.exp(pp).T)

plt.title(r'log post, num data = {}'.format(A.shape[0]))
axes.set_xlabel(r'$\alpha$')
axes.set_ylabel(r'$\beta$')
axes.set_zlabel('p')













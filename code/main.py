import os
import inspect
directory = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])) # get script's path
os.chdir(directory)
##

import numpy as np
from matplotlib import pyplot as plt
import scipy.special as spc
import scipy.stats as stat
from scipy import optimize
from scipy.integrate import simps as simpson  #from scipy.integrate import simpson #for new python versions
import numpy.random as rd
import math
import pickle
from numba import jit

from data import get_S_A
import reference_curves as ref
import stat_functions
from config import IM, C, path
from utils import rep0, rep1
from extract_saved_fisher import fisher_approx, jeffrey_approx
from distributions import log_gamma_normal_pdf


# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# # for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
# plt.ion()
# plt.show()




## fragility curve estimation via posterior estimation

S_tot, A_tot = get_S_A(path, IM, C, shuffle=False, relative=False)
num_tot = A_tot.shape[0]

num_est = 50
S_ref, A_ref = S_tot[:num_est], A_tot[:num_est]


# tirage de nombreux MLE

def func_log_vr(z, a) :
    def log_vr(theta) :
        return -stat_functions.log_vrais(z, a, theta.reshape(1,2))
        # return ref.opp_log_vraiss(theta, a, z)
    return log_vr

kmax = int(num_tot/num_est)
t0 = np.array([3,0.3])
th_MLE = np.zeros((kmax,2))
for k in range(kmax) :
    S = S_tot[k*num_est:(k+1)*num_est]
    A = A_tot[k*num_est:(k+1)*num_est]
    log_vr = func_log_vr(S,A)
    th_MLE[k] = optimize.minimize(log_vr, t0, bounds=[(0.01,100),(0.01,100)]).x


## tirage d'autant de th_post

def func_log_post(z,a) :
    @jit(nopython=True)
    def log_post(theta) :
        return stat_functions.log_post_jeff_adapt(theta,z,a, Fisher=fisher_approx)
    return log_post

# import time
# log_post = func_log_post(S_ref, A_ref)
# sig_prop = np.sqrt(0.4*np.array([[0.5,0],[0,0.2]]))
# a1 = time.time()
# th_post, th_tot, accept = stat_functions.HM_k(t0, log_post, kmax, pi_log=True, max_iter=1000, sigma_prop=sig_prop) #long time to execute ; prefer opening files from data bellow
# a2 = time.time()  # time for max_iter=kmax=2000 arround 16.5 minutes
# accept = np.minimum(accept, np.ones_like(accept))
#
# plt.figure()
# plt.plot(th_MLE[:,0], th_MLE[:,1], 'x', label='MLE')
# plt.plot(th_post[:,0], th_post[:,1], 'o', label='Post')
# plt.title(r'Tirages de {} MLE / obs. a posteriori'.format(kmax))
# plt.legend()

# file = open(path+r"/th_tot", mode='wb')   # To save the results of the Metropolis-Hasting execution
# pickle.dump(th_tot, file)
# file.close()
# file = open(path+r"/accept_ratio", mode='wb')
# pickle.dump(accept, file)
# file.close()
#
file = open(path+r'/th_tot', 'rb')   # To load the last results of the Metropolis-Hasting execution instead of executing it again
th_tot = pickle.load(file)           # instead of executing it again
th_post = th_tot[-1] + 0
file.close()
file = open(path+r'/accept_ratio', 'rb')
accept = pickle.load(file)
file.close()



## tracé des quantiles de confiance comparaison avec Ref curve

# ref_accept = 2.38*np.sqrt(2)

conf = 0.05

curves_post = 1/2+1/2*spc.erf(np.log(rep0(ref.a_tab, kmax)/rep1(th_post[:,0], ref.num_a))/rep1(th_post[:,1], ref.num_a))
curves_MLE = 1/2+1/2*spc.erf(np.log(rep0(ref.a_tab, kmax)/rep1(th_MLE[:,0], ref.num_a))/rep1(th_MLE[:,1], ref.num_a))

q1_post, q2_post = np.quantile(curves_post, 1-conf/2, axis=0), np.quantile(curves_post, conf/2, axis=0)
q1_MLE, q2_MLE = np.quantile(curves_MLE, 1-conf/2, axis=0), np.quantile(curves_MLE, conf/2, axis=0)

curve_q1_post, curve_q2_post = np.zeros(ref.num_a), np.zeros(ref.num_a)
curve_q1_MLE, curve_q2_MLE = np.zeros(ref.num_a), np.zeros(ref.num_a)

for i in range(ref.num_a) :
    curve_q1_post[i] = curves_post[np.abs(curves_post[:,i] - q1_post[i]).argmin()][i]
    curve_q2_post[i] = curves_post[np.abs(curves_post[:,i] - q2_post[i]).argmin()][i]
    curve_q1_MLE[i] = curves_MLE[np.abs(curves_MLE[:,i] - q1_MLE[i]).argmin()][i]
    curve_q2_MLE[i] = curves_MLE[np.abs(curves_MLE[:,i] - q2_MLE[i]).argmin()][i]

curve_med_post = np.median(curves_post, axis=0)
curve_med_MLE = np.median(curves_MLE, axis=0)



fig = plt.figure(figsize=(13,5))

ax1 = fig.add_subplot(1,2,1)
ax1.plot(ref.a_tab, curve_med_MLE, '-k', label=r'Median')
ax1.plot(ref.a_tab, curve_q1_MLE, '--r', label=r'conf {}%'.format(int((1-conf)*100)))
ax1.plot(ref.a_tab, curve_q2_MLE, '--r')
ax1.plot(A_ref, S_ref, 'og', alpha=0.6, fillstyle='none')
ax1.plot(ref.a_tab, ref.curve_ML, color='magenta', label=r'ref', alpha=0.7)
ax1.set_title(r'MLE estimation via bootstrap')
ax1.set_ylabel(r'$P_f(a)$')
ax1.set_xlabel(r'$a$='+IM)
ax1.legend()



ax2 = fig.add_subplot(1,2,2)
ax2.plot(ref.a_tab, curve_med_post, '-k', label=r'Median', linewidth=0.8)
ax2.plot(ref.a_tab, ref.curve_ML, color='magenta', label=r'ref', alpha=0.7, linewidth=0.8)
ax2.plot(ref.a_tab, curve_q1_post, '--r', label=r'conf {}%'.format(int((1-conf)*100)), linewidth=0.8)
ax2.plot(ref.a_tab, curve_q2_post, '--r', linewidth=0.8)
ax2.plot(A_ref, S_ref, 'og', alpha=0.6, fillstyle='none')

ax2.set_frame_on(False)
(xmin, xmax) = ax2.xaxis.get_view_interval()
(ymin, ymax) = ax2.yaxis.get_view_interval()
ax2.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax),
                              color = 'black', linewidth = 1.5))
ax2.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin),
                              color = 'black', linewidth = 1.5))
ax2.set_ylabel(r'$P_f(a)$')
ax2.set_xlabel(r'$a$='+IM)
ax2.legend()



box = ax2.get_position()
ax22 = fig.add_axes(np.concatenate((np.array([box.bounds[0], box.bounds[1]]), box.size)))
# ax22 = fig.add_subplot(1,2,2)
# ax2 = plt.gca().twinx()
ax22.xaxis.set_label_position('top')
ax22.yaxis.set_label_position('right')

ax22.plot(accept.mean(axis=1), '--b', label=r'accpet ratio', linewidth=0.5, alpha=0.3)
# ax22.plot(accept[:,0], '--g', label=r'accpet ratio', linewidth=0.5, alpha=0.3)

ax22.set_frame_on(False)
ax22.yaxis.tick_right()
ax22.xaxis.tick_top()
ax22.xaxis.set_tick_params(color = 'blue', labelcolor = 'blue')
ax22.yaxis.set_tick_params(color = 'blue', labelcolor = 'blue')
# ax2.xaxis.set_visible(False)
# ax22.set_ylim(0,1)
# (xmin, xmax) = ax2.xaxis.get_view_interval()
(ymin, ymax) = ax2.yaxis.get_view_interval()
ax22.set_ylim(ymin,ymax)
(xmin, xmax) = ax22.xaxis.get_view_interval()
(ymin, ymax) = ax22.yaxis.get_view_interval()
# ax22.plot([xmin,xmax],[ref_accept, ref_accept], color='cyan')
# ax22.set_yticks([ref_accept], minor = True)
# ax22.yaxis.set_ticklabels([r'${:4.2f}$'.format(ref_accept)], minor = True)
# ax22.yaxis.set_tick_params(which = 'minor', color = 'cyan', labelcolor = 'cyan')
# ax22.add_artist(plt.Line2D((xmin, xmax), (ref_accept, ref_accept), color = 'cyan', linewidth = 0.7))
ax22.add_artist(plt.Line2D((xmax, xmax), (ymin, ymax),
                              color = 'blue', linewidth = 1.5))
ax22.add_artist(plt.Line2D((xmin, xmax), (ymax, ymax),
                              color = 'blue', linewidth = 1.5))
ax22.set_ylabel(r'avg. accept ratio', color='blue')
ax22.set_xlabel(r'iterations', color='blue')
ax22.legend()


ax22.set_title(r'estimation with posterior')



## MCMC diagnosis

fig2 = plt.figure(figsize=(13,5))

ax1 = fig2.add_subplot(121)
ax1.plot(th_tot[:,0,0])
ax1.set_ylabel(r'$\alpha$')
ax1.set_xlabel(r'iterations')
ax1.set_title(r'Convergence of $\alpha$ in MCMC')

ax2 = fig2.add_subplot(122)
ax2.plot(th_tot[:,0,1])
ax2.set_ylabel(r'$\beta$')
ax2.set_xlabel(r'iterations')
ax2.set_title(r'Convergence of $\beta$ in MCMC')



fig3 = plt.figure(figsize=(13,5))
n_hist = int(kmax**(1/3))*5

ax1 = fig3.add_subplot(121)
ax1.hist(th_post[:,0], n_hist, density=True)
ax1.set_xlabel(r'$\alpha$')
ax1.set_title(r'$\alpha$ repartition after MCMC')

ax2 = fig3.add_subplot(122)
ax2.hist(th_post[:,1], n_hist, density=True)
ax2.set_xlabel(r'$\beta$')
ax2.set_title(r'$\beta$ repartition after MCMC')


## plot convergence


err_conf_post = simpson(np.abs(curve_q1_post - curve_q2_post)**2, ref.a_tab)**(1/2)
err_med_post = simpson(np.abs(curve_med_post - ref.curve_ML)**2, ref.a_tab)**(1/2)

err_conf_MLE = simpson(np.abs(curve_q1_MLE - curve_q2_MLE)**2, ref.a_tab)**(1/2)
err_med_MLE = simpson(np.abs(curve_med_MLE - ref.curve_ML)**2, ref.a_tab)**(1/2)

print('-- Comparison bootstrap MLE / simulation from posterior --')
print('prior = Jeffreys')
print('num_data = {}'.format(num_est))
print('num_simulations = {}'.format(kmax))
print()
print('err_conf post = {:8.2f}   ;   err_med post = {:8.2f}'.format(err_conf_post, err_med_post))
print('err_conf MLE  = {:8.2f}   ;   err_med MLE  = {:8.2f}'.format(err_conf_MLE, err_med_MLE))
print()




### Cnvergence comparison

# différences successives en fonction de du nombre de données considérées

gam_lamb = 0.01
gam_a = 0.01
gam_b = 0.01
gam_m = 0

def func_log_post_gamma(z,a, gam_a=gam_a, gam_b=gam_b, gam_m=gam_m, gam_lamb=gam_lamb) :
    @jit(nopython=True)
    def log_post(theta) :
        log_vrs = stat_functions.log_vrais(z, a, theta)
        log_prior = log_gamma_normal_pdf(np.log(theta[:,0]), theta[:,1], gam_m, gam_a, gam_b, gam_lamb)
        return log_prior + log_vrs
    return log_post

sigma_prop = np.sqrt(0.4*np.array([[0.5,0],[0,0.2]]))

num_est_max = 200
num_est_min = 1
number_estimations = num_est_max - num_est_min
num_est_tab = np.arange(num_est_min, num_est_max)
# kmax_conv = int(num_tot/num_est_max)+1
kmax_conv = 1000

iter_HM = 5000
keep_HM = 100
num_sim_HM = int(kmax_conv/keep_HM)
# assert keep_HM*num_sim_HM = kmax_conv

th_MLE_tab = np.zeros((number_estimations, kmax_conv, 2))
th_post_jeff_tab = np.zeros((number_estimations, kmax_conv, 2))
th_post_gam_tab = np.zeros((number_estimations, kmax_conv, 2))
accept_jeff_tab = np.zeros((number_estimations, iter_HM))
accept_gam_tab = np.zeros((number_estimations, iter_HM))

th_post_jeff_tot_tab = np.zeros((number_estimations, iter_HM, 2))
th_post_gam_tot_tab = np.zeros((number_estimations, iter_HM, 2))

print('starting convergence simulations')

for nn, num in enumerate(num_est_tab) :
    # calculate kmax th_MLE via bootstrap:
    # for k in range(kmax_conv) :
    #     i = rd.randint(0, num_tot-num_est_max)
    #     S = S_tot[i:i+num]+0
    #     A = A_tot[i:i+num]+0
    #     # print(S)
    #     # if len(S)==0 :
    #     #     print(i,num,k)
    #     #     break
    #     log_vr = func_log_vr(S,A)
        # th_MLE_tab[nn, k] = optimize.minimize(log_vr, t0, bounds=[(0.01,100),(0.01,100)], options={'maxiter':10, 'disp':False}).x
    th_MLE_tab[nn,:] = t0+0
    #simulate kmax_conv th_post_jeffrey via HM
    log_post = func_log_post(S_tot[:num], A_tot[:num])
    sigma_prop = np.array([[0.1,0],[0,0.095]])
    # t_fin, t_tot, acc = stat_functions.adaptative_HM_k(t0, log_post, num_sim_HM, pi_log=True, max_iter=iter_HM, sigma0=sigma_prop)
    t_fin, t_tot, acc = stat_functions.adaptative_HM(t0, log_post, pi_log=True, max_iter=20000, sigma0=sigma_prop)
    # th_post_jeff_tab[nn, :, 0] = t_tot[-keep_HM:, :, 0].flatten()
    # th_post_jeff_tab[nn, :, 1] = t_tot[-keep_HM:, :, 1].flatten()
    th_post_jeff_tab[nn, :, 0] = t_tot[-5000:, 0].flatten()
    th_post_jeff_tab[nn, :, 1] = t_tot[-5000:, 1].flatten()
    # accept_jeff_tab[nn] = np.minimum(acc,1).mean(axis=1)
    accept_jeff_tab[nn] = np.minimum(acc,1)
    # th_post_jeff_tot_tab[nn] = t_tot[:,0]+0
    th_post_jeff_tot_tab[nn] = t_tot[:]+0
    
    #simulate kmax_conv th_post_gamma via HM
    log_post = func_log_post_gamma(S_tot[:num], A_tot[:num])
    # sigma_prop =
    t_fin, t_tot, acc = stat_functions.adaptative_HM_k(t0, log_post, num_sim_HM, pi_log=True, max_iter=iter_HM)
    th_post_gam_tab[nn, :, 0] = t_tot[-keep_HM:, :, 0].flatten()
    th_post_gam_tab[nn, :, 1] = t_tot[-keep_HM:, :, 1].flatten()
    accept_gam_tab[nn] = np.minimum(acc,1).mean(axis=1)
    th_post_gam_tot_tab[nn] = t_tot[:,0]+0

    if nn%10==0 :
        print(r'{}/{}'.format(nn, number_estimations))



print('step 1 done' )

## # Compute med/conf curves and errors

conf = 0.05

curve_q1_jeff_tab = np.zeros((number_estimations, ref.num_a))
curve_q2_jeff_tab = np.zeros((number_estimations, ref.num_a))
curve_med_jeff_tab = np.zeros((number_estimations, ref.num_a))
curve_q1_gam_tab = np.zeros((number_estimations, ref.num_a))
curve_q2_gam_tab = np.zeros((number_estimations, ref.num_a))
curve_med_gam_tab = np.zeros((number_estimations, ref.num_a))
curve_q1_MLE_tab = np.zeros((number_estimations, ref.num_a))
curve_q2_MLE_tab = np.zeros((number_estimations, ref.num_a))
curve_med_MLE_tab = np.zeros((number_estimations, ref.num_a))

err_conf_jeff_tab = np.zeros(number_estimations)
err_med_jeff_tab = np.zeros(number_estimations)
err_conf_gam_tab = np.zeros(number_estimations)
err_med_gam_tab = np.zeros(number_estimations)
err_conf_MLE_tab = np.zeros(number_estimations)
err_med_MLE_tab = np.zeros(number_estimations)

for nn,num in enumerate(num_est_tab) :
    curves_jeff_tmp = 1/2+1/2*spc.erf(np.log(rep0(ref.a_tab, kmax_conv)/rep1(th_post_jeff_tab[nn,:,0], ref.num_a))/rep1(th_post_jeff_tab[nn,:,1], ref.num_a))
    curves_gam_tmp = 1/2+1/2*spc.erf(np.log(rep0(ref.a_tab, kmax_conv)/rep1(th_post_gam_tab[nn,:,0], ref.num_a))/rep1(th_post_gam_tab[nn,:,1], ref.num_a))
    curves_MLE_tmp = 1/2+1/2*spc.erf(np.log(rep0(ref.a_tab, kmax_conv)/rep1(th_MLE_tab[nn,:,0], ref.num_a))/rep1(th_MLE_tab[nn,:,1], ref.num_a))

    q1_jeff_tmp, q2_jeff_tmp = np.quantile(curves_jeff_tmp, 1-conf/2, axis=0), np.quantile(curves_jeff_tmp, conf/2, axis=0)
    q1_gam_tmp, q2_gam_tmp = np.quantile(curves_gam_tmp, 1-conf/2, axis=0), np.quantile(curves_gam_tmp, conf/2, axis=0)
    q1_MLE_tmp, q2_MLE_tmp = np.quantile(curves_MLE_tmp, 1-conf/2, axis=0), np.quantile(curves_MLE_tmp, conf/2, axis=0)

    for i in range(ref.num_a) :
        curve_q1_jeff_tab[nn,i] = curves_jeff_tmp[np.abs(curves_jeff_tmp[:,i] - q1_jeff_tmp[i]).argmin()][i]
        curve_q2_jeff_tab[nn,i] = curves_jeff_tmp[np.abs(curves_jeff_tmp[:,i] - q2_jeff_tmp[i]).argmin()][i]
        curve_q1_gam_tab[nn,i] = curves_gam_tmp[np.abs(curves_gam_tmp[:,i] - q1_gam_tmp[i]).argmin()][i]
        curve_q2_gam_tab[nn,i] = curves_gam_tmp[np.abs(curves_gam_tmp[:,i] - q2_gam_tmp[i]).argmin()][i]
        curve_q1_MLE_tab[nn,i] = curves_MLE_tmp[np.abs(curves_MLE_tmp[:,i] - q1_MLE_tmp[i]).argmin()][i]
        curve_q2_MLE_tab[nn,i] = curves_MLE_tmp[np.abs(curves_MLE_tmp[:,i] - q2_MLE_tmp[i]).argmin()][i]

    curve_med_jeff_tab[nn] = np.median(curves_jeff_tmp, axis=0)
    curve_med_gam_tab[nn] = np.median(curves_gam_tmp, axis=0)
    curve_med_MLE_tab[nn] = np.median(curves_MLE_tmp, axis=0)

    err_conf_jeff_tab[nn] = simpson(np.abs(curve_q1_jeff_tab[nn] - curve_q2_jeff_tab[nn])**2, ref.a_tab)**(1/2)
    err_med_jeff_tab[nn] = simpson(np.abs(curve_med_jeff_tab[nn] - ref.curve_ML)**2, ref.a_tab)**(1/2)
    err_conf_gam_tab[nn] = simpson(np.abs(curve_q1_gam_tab[nn] - curve_q2_gam_tab[nn])**2, ref.a_tab)**(1/2)
    err_med_gam_tab[nn] = simpson(np.abs(curve_med_gam_tab[nn] - ref.curve_ML)**2, ref.a_tab)**(1/2)
    err_conf_MLE_tab[nn] = simpson(np.abs(curve_q1_MLE_tab[nn] - curve_q2_MLE_tab[nn])**2, ref.a_tab)**(1/2)
    err_med_MLE_tab[nn] = simpson(np.abs(curve_med_MLE_tab[nn] - ref.curve_ML)**2, ref.a_tab)**(1/2)


#todo : think about a sigma_prop

## plots

#plot conv curves

fig4 = plt.figure(figsize=(13,5))

ax1 = fig4.add_subplot(121)
ax1.plot(err_conf_MLE_tab, label=r'MLE')
ax1.plot(err_conf_jeff_tab, label=r'Jeff')
ax1.plot(err_conf_gam_tab, label=r'Gamma')
ax1.set_title(r'Confidence scale: $|q_{r/2}-q_{1-r/2}|_2$, '+r'$r={}$'.format(conf))
ax1.legend()
ax1.set_xlabel(r'Number of data')
ax1.set_ylabel(r'error')

ax2 = fig4.add_subplot(122)
ax2.plot(err_med_MLE_tab, label=r'MLE')
ax2.plot(err_med_jeff_tab, label=r'Jeff')
ax2.plot(err_med_gam_tab, label=r'Gamma')
ax2.set_title(r'Median error $|q_{med} - P_{ref}|_2$')
ax2.legend()
ax2.set_xlabel(r'Number of data')
ax2.set_ylabel(r'error')


# plot some curves computed during process


## HM diagnosis

# plot diagnosis similar than above, for a few steps in the range of the data
# scatter plots, histograms, acceptance rates, raw HM updates

num_plots = 3
step_plots = int(number_estimations/num_plots)
tab_nn = [(i+1)*step_plots for i in range(num_plots)]

colors = ['blue', 'orange', 'green', 'red', 'magenta', 'grey', 'yellow']

fig51, axes51 = plt.subplots(2,3) #sacterplots + curves 2x3
fig52, axes52 = plt.subplots(2,2) #histograms 2x2
fig53, axes53 = plt.subplots(1,2) #accept rates 1x2 accept rates on
fig54, axes54 = plt.subplots(2,2) #raw HMs 2x2
#todo : later, accept rates on curves, raw HMs on hists

# ax1 = fig5.add_subplot(221)
# ax2 = fig5.add_subplot(222)
# ax3 = fig5.add_subplot(223)
# ax4 = fig5.add_subplot(224)

axes51[0,0].set_title(r'MLE')
axes51[0,1].set_title(r'Jeffreys prior')
axes51[0,2].set_title(r'Gamma-normal prior')
axes51[1,0].set_title(r'MLE')
axes51[1,1].set_title(r'Jeffreys prior')
axes51[1,2].set_title(r'Gamma-normal prior')

axes52[0,0].set_title(r'Jeffreys prior')
axes52[0,1].set_title(r'Gamma-normal prior')
axes52[1,0].set_title(r'Jeffreys prior')
axes52[1,1].set_title(r'Gamma-normal prior')

axes53[0].set_title(r'Jeffreys prior')
axes53[1].set_title(r'Gamma-normal prior')

axes54[0,0].set_title(r'Jeffreys prior')
axes54[0,1].set_title(r'Gamma-normal prior')
axes54[1,0].set_title(r'Jeffreys prior')
axes54[1,1].set_title(r'Gamma-normal prior')

num_hist = 40
arr_alp_hist, h_alp_hist = np.linspace(2,8, num=num_hist, retstep=True)
arr_bet_hist, h_bet_hist = np.linspace(0.2,1.5, num=num_hist, retstep=True)

for i,nn in enumerate(tab_nn) :
    num = num_est_tab[nn]

    axes51[0,0].plot(th_MLE_tab[nn,:,0], th_MLE_tab[nn,:,1], 'o', color=colors[i], label=r'{}'.format(num), markersize=3)
    axes51[0,1].plot(th_post_jeff_tab[nn,:,0], th_post_jeff_tab[nn,:,1], 'o', color=colors[i], label=r'{}'.format(num), markersize=3)
    axes51[0,2].plot(th_post_gam_tab[nn,:,0], th_post_gam_tab[nn,:,1], 'o', color=colors[i], label=r'{}'.format(num), markersize=3)

    axes51[1,0].plot(ref.a_tab, curve_q1_MLE_tab[nn], label=r'{}'.format(num), color=colors[i])
    axes51[1,0].fill_between(ref.a_tab, curve_q1_MLE_tab[nn], curve_q2_MLE_tab[nn], facecolor=colors[i])
    axes51[1,1].plot(ref.a_tab, curve_q1_jeff_tab[nn], label=r'{}'.format(num), color=colors[i])
    axes51[1,1].fill_between(ref.a_tab, curve_q1_jeff_tab[nn], curve_q2_jeff_tab[nn], facecolor=colors[i])
    axes51[1,2].plot(ref.a_tab, curve_q1_gam_tab[nn], label=r'{}'.format(num), color=colors[i])
    axes51[1,2].fill_between(ref.a_tab, curve_q1_gam_tab[nn], curve_q2_gam_tab[nn], facecolor=colors[i])

    axes52[0,0].hist(th_post_jeff_tab[nn,:,0], bins=arr_alp_hist+i*h_alp_hist/num_plots, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i], density=True)
    axes52[0,1].hist(th_post_gam_tab[nn,:,0], bins=arr_alp_hist+i*h_alp_hist/num_plots, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i], density=True)
    axes52[1,0].hist(th_post_jeff_tab[nn,:,1], bins=arr_bet_hist+i*h_bet_hist/num_plots, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i], density=True)
    axes52[1,1].hist(th_post_gam_tab[nn,:,1], bins=arr_bet_hist+i*h_bet_hist/num_plots, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i], density=True)

    axes53[0].plot(accept_jeff_tab[nn].cumsum()/np.arange(1,20000)[10*np.range(2000)], color=colors[i], label=r'{}'.format(num), linewidth=0.5, alpha=1)
    #axes53[1].plot(accept_gam_tab[nn], color=colors[i], label=r'{}'.format(num), linewidth=0.5, alpha=1)

    axes54[0,0].plot(th_post_jeff_tot_tab[nn,:,0], color=colors[i], label=r'{}'.format(num), linewidth=1, alpha=0.7)
    axes54[0,1].plot(th_post_gam_tot_tab[nn,:,0], color=colors[i], label=r'{}'.format(num), linewidth=1, alpha=0.7)
    axes54[1,0].plot(th_post_jeff_tot_tab[nn,:,1], color=colors[i], label=r'{}'.format(num), linewidth=1, alpha=0.7)
    axes54[1,1].plot(th_post_gam_tot_tab[nn,:,1], color=colors[i], label=r'{}'.format(num), linewidth=1, alpha=0.7)
    # ax2.hist()
    # ax3.plot(np.minimum(accept_jeff_tab[nn],1), color=colors[i], label=r'{}'.format(num), linewidth=0.5, alpha=1)
    # ax4.plot(th_post_jeff_tot_tab[nn,:,0], color=colors[i], label=r'{}'.format(num), linewidth=1, alpha=0.7)

# ax1.set_xlabel(r'$\alpha$')
# ax1.set_ylabel(r'$\beta$')
# ax1.set_title(r'Jeff prior, scatter plot')
# ax1.legend()
# # ax2.l
# ax3.legend()
# ax4.legend()

for i in range(3) :
    axes51[1,i].plot(ref.a_tab, ref.curve_ML, color='magenta', label=r'ref')


for ax in axes51.flatten():
    ax.legend()
for ax in axes52.flatten():
    ax.legend()
for ax in axes53:
    ax.legend()
for ax in axes54.flatten():
    ax.legend()

for ax in axes51[0] :
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
for ax in axes51[1] :
    ax.set_xlabel(r'$a=$'+IM)
    ax.set_ylabel(r'$P_f(a)$')
for ax in axes53 :
    ax.set_xlabel(r'$iterations$')
    ax.set_ylabel(r'avg. accept ratio')
for ax in axes54[0] :
    ax.set_xlabel(r'iterations')
    ax.set_ylabel(r'$\alpha$ evolution')
for ax in axes54[1] :
    ax.set_xlabel(r'iterations')
    ax.set_ylabel(r'$\beta$ evolution')



### make posterior density curves to compare with hists

n_alp = 100
n_bet = 100

alpha_array, h_alp = np.linspace(2,8, num=n_alp, retstep=True)
beta_array, h_bet = np.linspace(0.2,1.5, num=n_bet, retstep=True)
al_grid, be_grid = np.meshgrid(alpha_array, beta_array)
# th_grid = np.zeros((n_alp,n_bet,2))
th_grid = np.concatenate((al_grid[...,np.newaxis], be_grid[...,np.newaxis]), axis=-1)

vrais = np.zeros((number_estimations, n_alp, n_bet))
for nn, num in enumerate(num_est_tab) :
    for i in range(n_alp) :
        vrais[nn,i] = stat_functions.log_vrais(S_tot[:num], A_tot[:num], th_grid[i])
    if nn%10==0 :
        print(r'{}/{}'.format(nn, number_estimations))

th_grid2 = th_grid+0
th_grid2[:,:,0] = np.log(th_grid[:,:,0])

post_jeff_grid = np.zeros((number_estimations, n_alp, n_bet))
post_gam_grid = np.zeros((number_estimations, n_alp, n_bet))
for i in range(n_alp) :
    post_jeff_grid[:,i] = rep0(np.log(jeffrey_approx(th_grid[i])), number_estimations)
    post_gam_grid[:,i] = rep0(log_gamma_normal_pdf(np.log(th_grid[i,:,0]), th_grid[i,:,1], gam_m, gam_a, gam_b, gam_lamb), number_estimations)

post_jeff_grid = post_jeff_grid + vrais
post_gam_grid = post_gam_grid + vrais
post_jeff_grid = np.exp(post_jeff_grid)
post_gam_grid = np.exp(post_gam_grid)



## plot the hists

for i,nn in enumerate(tab_nn) :
    num = num_est_tab[nn]

    # axes52[0,0].hist(th_post_jeff_tab[nn,:,0]+i*h_alp, bins=alpha_array, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i])
    # axes52[0,1].hist(th_post_gam_tab[nn,:,0]+i*h_alp, bins=alpha_array, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i])
    # axes52[1,0].hist(th_post_jeff_tab[nn,:,1]+i*h_bet, bins=beta_array, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i])
    # axes52[1,1].hist(th_post_jeff_tab[nn,:,1]+i*h_bet, bins=beta_array, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i])

    #alpha_density, then beta_density :
    alpha_density_jeff = simpson(post_jeff_grid[nn], beta_array, axis=-1)
    Kalp = simpson(alpha_density_jeff, alpha_array)
    beta_density_jeff = simpson(post_jeff_grid[nn], alpha_array, axis=-2)
    Kbet = simpson(beta_density_jeff, beta_array)
    axes52[0,0].plot(alpha_array, alpha_density_jeff/Kalp, color=colors[i])
    axes52[1,0].plot(beta_array, beta_density_jeff/Kbet, color= colors[i])

    alpha_density_gam = simpson(post_gam_grid[nn], beta_array, axis=-1)
    Kalp = simpson(alpha_density_gam, alpha_array)
    beta_density_gam = simpson(post_gam_grid[nn], alpha_array, axis=-2)
    Kbet = simpson(beta_density_gam, beta_array)
    axes52[0,1].plot(alpha_array, alpha_density_gam/Kalp, color=colors[i])
    axes52[1,1].plot(beta_array, beta_density_gam/Kbet, color= colors[i])

for ax in axes52[0,:] :
    ax.set_xlim(2,6)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'post$(\alpha)$')
for ax in axes52[1,:] :
    ax.set_xlim(0.23,1)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'post$(\beta)$')



## add cumulative expected value on fragility curves

cl_colors = ['cyan', 'gold', 'lightgreen', 'salmon', 'violet', 'lightgrey', 'lightyellow']

axes51bis = []

for ax in axes51[1,1:] :
    ax.set_frame_on(False)
    (xmin, xmax) = ax.xaxis.get_view_interval()
    (ymin, ymax) = ax.yaxis.get_view_interval()
    ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax),
                                color = 'black', linewidth = 1.5))
    ax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin),
                                color = 'black', linewidth = 1.5))

    box = ax.get_position()
    axbis = fig.add_axes(np.concatenate((np.array([box.bounds[0], box.bounds[1]]), box.size)))
    axbis.xaxis.set_label_position('top')
    axbis.yaxis.set_label_position('right')

    axbis.set_frame_on(False)
    axbis.yaxis.tick_right()
    axbis.xaxis.tick_top()
    axbis.xaxis.set_tick_params(color = 'blue', labelcolor = 'blue')
    axbis.yaxis.set_tick_params(color = 'blue', labelcolor = 'blue')

    axbis.set_ylabel(r'$\mathbb{E}[\beta]$  $\mathbb{E}[\alpha]$', color='blue')
    axbis.set_xlabel(r'iterations', color='blue')

    axes51bis.append(axbis)

    # ax22.set_title(r'estimation with posterior')


num_kept_HM_fin = th_post_jeff_tot_tab[nn,:,0].shape[0]
for i,nn in enumerate(tab_nn) :
    num = num_est_tab[nn]

    axes51bis[0].plot(th_post_jeff_tot_tab[nn,:,0].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i])
    axes51bis[1].plot(th_post_gam_tot_tab[nn,:,0].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i])


    (ymin, ymax) = ax.yaxis.get_view_interval()
    axbis.set_ylim(ymin,ymax)
    (xmin, xmax) = axbis.xaxis.get_view_interval()
    (ymin, ymax) = axbis.yaxis.get_view_interval()
    axbis.add_artist(plt.Line2D((xmax, xmax), (ymin, ymax),
                              color = 'blue', linewidth = 1.5))
    axbis.add_artist(plt.Line2D((xmin, xmax), (ymax, ymax),
                              color = 'blue', linewidth = 1.5))



    axbis.legend()








import os
import inspect
directory = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])) # get script's path
os.chdir(directory)
##

import numpy as np
from matplotlib import pyplot as plt
import scipy.special as spc
from scipy import optimize
from scipy.integrate import simps as simpson  #from scipy.integrate import simpson #for new python versions
import numpy.random as rd
import math
# import pickle
from numba import jit

from data import get_S_A
import reference_curves as ref
import stat_functions
from config import IM, C, path, save_fisher_arr
from utils import rep0, rep1
from extract_saved_fisher import fisher_approx, jeffrey_approx
from distributions import log_gamma_normal_pdf

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.ion()
plt.show()

##

S_tot, A_tot = get_S_A(path, IM, C, shuffle=False, relative=False)
num_tot = A_tot.shape[0]


##
## 1. Simulate for nn in tab_nn



num_est_max = 200
num_est_min = 1
number_estimations = num_est_max - num_est_min
num_est_tab = np.arange(num_est_min, num_est_max)
# kmax_conv = int(num_tot/num_est_max)+1
#kmax_conv = 1000
# kmax_conv=5000
k_maxMLE = 100 # 500
k_maxHM = 10000

iter_HM = 40000
keep_HM = k_maxHM
num_sim_HM = 1



num_plots = 3
#step_plots = int(number_estimations/num_plots)
#tab_nn = [(i+1)*step_plots for i in range(num_plots)]
tab_nn = [50,100,150]

colors = ['blue', 'orange', 'green', 'red', 'magenta', 'grey', 'yellow']




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


def func_log_vr(z, a) :
    def log_vr(theta) :
        return -stat_functions.log_vrais(z, a, theta.reshape(1,2))
        # return ref.opp_log_vraiss(theta, a, z)
    return log_vr

def func_log_post(z,a) :
    @jit(nopython=True)
    def log_post(theta) :
        return stat_functions.log_post_jeff_adapt(theta,z,a, Fisher=fisher_approx)
    return log_post

# sigma_prop = np.sqrt(0.4*np.array([[0.5,0],[0,0.2]]))


## Simulations

t0 = np.array([3,0.3])


th_MLE_tab = np.zeros((num_plots, k_maxMLE, 2))
th_post_jeff_tab = np.zeros((num_plots, k_maxHM, 2))
th_post_gam_tab = np.zeros((num_plots, k_maxHM, 2))
accept_jeff_tab = np.zeros((num_plots, iter_HM))
accept_gam_tab = np.zeros((num_plots, iter_HM))

th_post_jeff_tot_tab = np.zeros((num_plots, iter_HM, 2))
th_post_gam_tot_tab = np.zeros((num_plots, iter_HM, 2))

print('starting convergence simulations')

for nn, num in enumerate(tab_nn) :
    # calculate kmax th_MLE via bootstrap:
    # CHANGE: REEL BOOTSTRAP
    # 1. Create A density conditionnaly
    sig1 = A_tot[:num][(S_tot[:num]==1)].var()**0.5
    h1 = sig1*(S_tot[:num]==1).sum()**(-1/5)
    sig2 = A_tot[:num][(S_tot[:num]==0)].var()**0.5
    h2 = sig2*(S_tot[:num]==0).sum()**(-1/5)
    @jit
    def f_A_cond_1(at) :
        # if np.all(at<=0) :
        #     return np.array([10**-15])
        # else :
        return np.array(np.exp(-((A_tot[:num][(S_tot[:num]==1)]-at)/h1)**2 ).mean()/h1/np.sqrt(np.pi)) * (at>0) + 10**-15*(at<=0)
    @jit
    def f_A_cond_2(at) :
        # if np.all(at<=0) :
        #     return np.array([10**-15])
        return np.array(np.exp(-((A_tot[:num][(S_tot[:num]==0)]-at)/h1)**2 ).mean()/h2/np.sqrt(np.pi)) * (at>0) + 10**-15*(at<=0)
    nb1 = (rd.rand(100,num)<((S_tot[:num]==1).mean())).sum()
    nb2 = num*100-nb1
    atinit1 = A_tot[:num][(S_tot[:num]==1)].mean()
    at_fin1, at_tot1, at_acc1 =  stat_functions.adaptative_HM_1d(np.array([atinit1]), f_A_cond_1, pi_log=False, max_iter=40000)
    A1 = at_tot1[-nb1:]
    atinit2 = A_tot[:num][(S_tot[:num]==0)].mean()
    at_fin2, at_tot2, at_acc2 =  stat_functions.adaptative_HM_1d(np.array([atinit2]), f_A_cond_2, pi_log=False, max_iter=40000)
    A2 = at_tot2[-nb2:]
    A = np.concatenate((A1,A2))
    S = np.ones(num*100)
    S[-nb2:] = 0
    id_sh = np.arange(num*100)
    rd.shuffle(id_sh)
    A = A[id_sh] +0
    S = S[id_sh] +0

    for k in range(k_maxMLE) :
        i = rd.randint(0, num*99)
        Stmp = S[i:i+num]+0
        Atmp = A[i:i+num]+0
        log_vr = func_log_vr(Stmp,Atmp)
        th_MLE_tab[nn, k] = optimize.minimize(log_vr, t0, bounds=[(0.01,100),(0.01,100)], options={'maxiter':10, 'disp':False}).x

    #simulate kmax th_post_jeffrey via HM
    log_post = func_log_post(S_tot[:num], A_tot[:num])
    sigma_prop = np.array([[0.1,0],[0,0.095]])
    t_fin, t_tot, acc = stat_functions.adaptative_HM(t0, log_post, pi_log=True, max_iter=iter_HM, sigma0=sigma_prop)
    th_post_jeff_tab[nn, :, 0] = t_tot[-keep_HM:, 0]
    th_post_jeff_tab[nn, :, 1] = t_tot[-keep_HM:, 1]
    accept_jeff_tab[nn] = np.minimum(acc,1)
    th_post_jeff_tot_tab[nn] = t_tot[:]+0

    #simulate kmax th_post_gamma via HM
    log_post = func_log_post_gamma(S_tot[:num], A_tot[:num])
    # sigma_prop =
    t_fin, t_tot, acc =  stat_functions.adaptative_HM(t0, log_post, pi_log=True, max_iter=iter_HM, sigma0=sigma_prop)
    th_post_gam_tab[nn, :, 0] = t_tot[-keep_HM:, 0]
    th_post_gam_tab[nn, :, 1] = t_tot[-keep_HM:, 1]
    accept_gam_tab[nn] = np.minimum(acc,1)
    th_post_gam_tot_tab[nn] = t_tot[:]+0

    # if nn%10==0 :
    #     print(r'{}/{}'.format(nn, number_estimations))








## confidence intervals and meds


conf = 0.05

curve_q1_jeff_tab = np.zeros((num_plots, ref.num_a))
curve_q2_jeff_tab = np.zeros((num_plots, ref.num_a))
curve_med_jeff_tab = np.zeros((num_plots, ref.num_a))
curve_q1_gam_tab = np.zeros((num_plots, ref.num_a))
curve_q2_gam_tab = np.zeros((num_plots, ref.num_a))
curve_med_gam_tab = np.zeros((num_plots, ref.num_a))
curve_q1_MLE_tab = np.zeros((num_plots, ref.num_a))
curve_q2_MLE_tab = np.zeros((num_plots, ref.num_a))
curve_med_MLE_tab = np.zeros((num_plots, ref.num_a))

err_conf_jeff_tab = np.zeros(num_plots)
err_med_jeff_tab = np.zeros(num_plots)
err_conf_gam_tab = np.zeros(num_plots)
err_med_gam_tab = np.zeros(num_plots)
err_conf_MLE_tab = np.zeros(num_plots)
err_med_MLE_tab = np.zeros(num_plots)

for nn,num in enumerate(tab_nn) :
    curves_jeff_tmp = 1/2+1/2*spc.erf(np.log(rep0(ref.a_tab, k_maxHM)/rep1(th_post_jeff_tab[nn,:,0], ref.num_a))/rep1(th_post_jeff_tab[nn,:,1], ref.num_a))
    curves_gam_tmp = 1/2+1/2*spc.erf(np.log(rep0(ref.a_tab, k_maxHM)/rep1(th_post_gam_tab[nn,:,0], ref.num_a))/rep1(th_post_gam_tab[nn,:,1], ref.num_a))
    curves_MLE_tmp = 1/2+1/2*spc.erf(np.log(rep0(ref.a_tab, k_maxMLE)/rep1(th_MLE_tab[nn,:,0], ref.num_a))/rep1(th_MLE_tab[nn,:,1], ref.num_a))

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




## HM diagnosis

# plot diagnosis similar than above, for a few steps in the range of the data
# scatter plots, histograms, acceptance rates, raw HM updates



fig51, axes51 = plt.subplots(2,3) #sacterplots + curves 2x3
fig52, axes52 = plt.subplots(2,2) #histograms 2x2
fig53, axes53 = plt.subplots(1,2) #accept rates 1x2 accept rates on
fig54, axes54 = plt.subplots(2,2) #raw HMs 2x2
#todo : later, accept rates on curves, raw HMs on hists


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

for i,nnn in enumerate(tab_nn) :
    num = num_est_tab[nnn]
    nn = i

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

    id_kp = int(iter_HM/2000)
    axes53[0].plot(id_kp*np.arange(2000), (accept_jeff_tab[nn].cumsum()/np.arange(1,iter_HM+1))[id_kp*np.arange(2000)], color=colors[i], label=r'{}'.format(num), linewidth=0.5, alpha=1)
    axes53[1].plot(id_kp*np.arange(2000), (accept_gam_tab[nn].cumsum()/np.arange(1,iter_HM+1))[id_kp*np.arange(2000)], color=colors[i], label=r'{}'.format(num), linewidth=0.5, alpha=1)

    axes54[0,0].plot(np.arange(iter_HM)[-2000:], th_post_jeff_tot_tab[nn,-2000:,0], color=colors[i], label=r'{}'.format(num), linewidth=1, alpha=0.7)
    axes54[0,1].plot(np.arange(iter_HM)[-2000:], th_post_gam_tot_tab[nn,-2000:,0], color=colors[i], label=r'{}'.format(num), linewidth=1, alpha=0.7)
    axes54[1,0].plot(np.arange(iter_HM)[-2000:], th_post_jeff_tot_tab[nn,-2000:,1], color=colors[i], label=r'{}'.format(num), linewidth=1, alpha=0.7)
    axes54[1,1].plot(np.arange(iter_HM)[-2000:], th_post_gam_tot_tab[nn,-2000:,1], color=colors[i], label=r'{}'.format(num), linewidth=1, alpha=0.7)




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
    ax.set_ylim(0,1)
for ax in axes54[0] :
    ax.set_xlabel(r'iterations')
    ax.set_ylabel(r'$\alpha$ evolution')
for ax in axes54[1] :
    ax.set_xlabel(r'iterations')
    ax.set_ylabel(r'$\beta$ evolution')



### make posterior density curves to compare with hists

n_alp = 100
n_bet = 100
# n_alp = save_fisher_arr.num_alpha
# n_bet = save_fisher_arr.num_beta

alpha_array, h_alp = np.linspace(2,8, num=n_alp, retstep=True)
beta_array, h_bet = np.linspace(0.2,1.5, num=n_bet, retstep=True)
# alpha_array, h_alp = np.linspace(1.5,8, num=n_alp, retstep=True)
# beta_array, h_bet = np.linspace(0.2,2, num=n_bet, retstep=True)
# alpha_array = save_fisher_arr.alpha_tab
# h_alp = save_fisher_arr.h_alpha
# beta_array = save_fisher_arr.beta_tab
# h_bet = save_fisher_arr.h_beta
al_grid, be_grid = np.meshgrid(alpha_array, beta_array)
# th_grid = np.zeros((n_alp,n_bet,2))
th_grid = np.concatenate((al_grid[...,np.newaxis], be_grid[...,np.newaxis]), axis=-1)

vrais = np.zeros((num_plots, n_alp, n_bet))
for nn, num in enumerate(tab_nn) :
    for i in range(n_alp) :
        vrais[nn,i] = stat_functions.log_vrais(S_tot[:num], A_tot[:num], th_grid[i])
    # if nn%10==0 :
    #     print(r'{}/{}'.format(nn, num_plots))

th_grid2 = th_grid+0
th_grid2[:,:,0] = np.log(th_grid[:,:,0])

post_jeff_grid = np.zeros((num_plots, n_alp, n_bet))
post_gam_grid = np.zeros((num_plots, n_alp, n_bet))
for i in range(n_alp) :
    post_jeff_grid[:,i] = rep0(np.log(jeffrey_approx(th_grid[i])), num_plots)
    post_gam_grid[:,i] = rep0(log_gamma_normal_pdf(np.log(th_grid[i,:,0]), th_grid[i,:,1], gam_m, gam_a, gam_b, gam_lamb), num_plots)

post_jeff_grid = post_jeff_grid + vrais
post_gam_grid = post_gam_grid + vrais
post_jeff_grid = np.exp(post_jeff_grid)
post_gam_grid = np.exp(post_gam_grid)


###
#
# j_min, j_max = 0, np.max(post_jeff_grid[0])
# levels = np.linspace(j_min, j_max, 15)
#
# plt.figure(figsize=(4.5, 2.5))
# plt.contourf(alpha_array, beta_array, post_jeff_grid[0], cmap='viridis', levels=levels)
# plt.title(r'posterior with Jeff prior, num data = {}'.format(A.shape[0]))
# plt.axis([1.5, 6, 0.2, 0.8])
# plt.colorbar()
# plt.xlabel(r"$\alpha$")
# plt.ylabel(r"$\beta$")
# plt.tight_layout()
# plt.show()



## plot the hists

for i,nnn in enumerate(tab_nn) :
    num = num_est_tab[nnn]
    nn = i

    # axes52[0,0].hist(th_post_jeff_tab[nn,:,0]+i*h_alp, bins=alpha_array, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i])
    # axes52[0,1].hist(th_post_gam_tab[nn,:,0]+i*h_alp, bins=alpha_array, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i])
    # axes52[1,0].hist(th_post_jeff_tab[nn,:,1]+i*h_bet, bins=beta_array, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i])
    # axes52[1,1].hist(th_post_jeff_tab[nn,:,1]+i*h_bet, bins=beta_array, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i])

    #alpha_density, then beta_density :
    # alpha_density_jeff = simpson(post_jeff_grid[nn], beta_array, axis=-1)
    alpha_density_jeff = post_jeff_grid[nn].sum(axis=-1)
    Kalp = simpson(alpha_density_jeff, alpha_array)
    # beta_density_jeff = simpson(post_jeff_grid[nn], alpha_array, axis=-2)
    beta_density_jeff = post_jeff_grid[nn].sum(axis=0)
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

axes52bis = []

for axx in axes52.flatten() :
    axx.set_frame_on(False)
    (xmin, xmax) = axx.xaxis.get_view_interval()
    (ymin, ymax) = axx.yaxis.get_view_interval()
    axx.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax),
                                color = 'black', linewidth = 1.5))
    axx.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin),
                                color = 'black', linewidth = 1.5))

    box = axx.get_position()
    axbis = fig52.add_axes(np.concatenate((np.array([box.bounds[0], box.bounds[1]]), box.size)))
    axbis.xaxis.set_label_position('top')
    axbis.yaxis.set_label_position('right')

    axbis.set_frame_on(False)
    axbis.yaxis.tick_right()
    axbis.xaxis.tick_top()
    axbis.xaxis.set_tick_params(color = 'blue', labelcolor = 'blue')
    axbis.yaxis.set_tick_params(color = 'blue', labelcolor = 'blue')

    # axbis.set_ylabel(r'$E[\beta]$  $E[\alpha]$', color='blue')
    axbis.set_xlabel(r'iterations', color='blue')

    axes52bis.append(axbis)

    # ax22.set_title(r'estimation with posterior')


num_kept_HM_fin = th_post_jeff_tot_tab[nn,:,0].shape[0]
for i,nnn in enumerate(tab_nn) :
    num = num_est_tab[nnn]
    nn = i

    axes52bis[0].plot(th_post_jeff_tot_tab[nn,:,0].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))
    axes52bis[1].plot(th_post_gam_tot_tab[nn,:,0].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))
    axes52bis[2].plot(th_post_jeff_tot_tab[nn,:,1].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))
    axes52bis[3].plot(th_post_gam_tot_tab[nn,:,1].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))

for j, axbis in enumerate(axes52bis) :
    # (ymin, ymax) = axbis.yaxis.get_view_interval()
    # axbis.set_ylim(ymin,ymax)
    (xmin, xmax) = axbis.xaxis.get_view_interval()
    (ymin, ymax) = axbis.yaxis.get_view_interval()
    axbis.add_artist(plt.Line2D((xmax, xmax), (ymin, ymax),
                              color = 'blue', linewidth = 1.5))
    axbis.add_artist(plt.Line2D((xmin, xmax), (ymax, ymax),
                              color = 'blue', linewidth = 1.5))

    if j<2 :
        axbis.set_ylabel(r'$E[\alpha]$', color='blue')
    else :
        axbis.set_ylabel(r'$E[\beta]$', color='blue')


    axbis.legend()




## add rates on fragility curves

axes51bis = []

for axx in axes51[1,1:] :
    axx.set_frame_on(False)
    (xmin, xmax) = axx.xaxis.get_view_interval()
    (ymin, ymax) = axx.yaxis.get_view_interval()
    axx.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax),
                                color = 'black', linewidth = 1.5))
    axx.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin),
                                color = 'black', linewidth = 1.5))

    box = axx.get_position()
    axbis = fig51.add_axes(np.concatenate((np.array([box.bounds[0], box.bounds[1]]), box.size)))
    axbis.xaxis.set_label_position('top')
    axbis.yaxis.set_label_position('right')
    axbis.set_ylim(ymin,ymax)

    axbis.set_frame_on(False)
    axbis.yaxis.tick_right()
    axbis.xaxis.tick_top()
    axbis.xaxis.set_tick_params(color = 'blue', labelcolor = 'blue')
    axbis.yaxis.set_tick_params(color = 'blue', labelcolor = 'blue')

    # axbis.set_ylabel(r'$E[\beta]$  $E[\alpha]$', color='blue')
    axbis.set_xlabel(r'iterations', color='blue')
    axbis.set_ylabel(r'accept rate', color='blue')

    axes51bis.append(axbis)

    # ax22.set_title(r'estimation with posterior')


num_kept_HM_fin = th_post_jeff_tot_tab[nn,:,0].shape[0]
for i,nnn in enumerate(tab_nn) :
    num = num_est_tab[nnn]
    nn = i

    axes51bis[0].plot(id_kp*np.arange(2000), (accept_jeff_tab[nn].cumsum()/np.arange(1,iter_HM+1))[id_kp*np.arange(2000)], '--', color=cl_colors[i], label=r'{}'.format(num), linewidth=0.8, alpha=1)
    axes51bis[1].plot(id_kp*np.arange(2000), (accept_gam_tab[nn].cumsum()/np.arange(1,iter_HM+1))[id_kp*np.arange(2000)], '--', color=cl_colors[i], label=r'{}'.format(num), linewidth=0.8, alpha=1)


for j, axbis in enumerate(axes51bis) :
    # (ymin, ymax) = axbis.yaxis.get_view_interval()
    # axbis.set_ylim(ymin,ymax)
    (xmin, xmax) = axbis.xaxis.get_view_interval()
    (ymin, ymax) = axbis.yaxis.get_view_interval()
    axbis.add_artist(plt.Line2D((xmax, xmax), (ymin, ymax),
                              color = 'blue', linewidth = 1.5))
    axbis.add_artist(plt.Line2D((xmin, xmax), (ymax, ymax),
                              color = 'blue', linewidth = 1.5))

    #axbis.legend()




# ## plot contour of post on scatter plots
#
# for i,nnn in enumerate(tab_nn) :
#     num = num_est_tab[nnn]
#     nn = i
#
#     axes51[0,1].contour(al_grid, be_grid, post_jeff_grid)












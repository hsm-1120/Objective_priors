# Figures d'illustrations

import os
import inspect
directory = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])) # get script's path
os.chdir(directory)

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

from config import path, IM, C
import reference_curves as ref
from data import get_data, get_S_A
from extract_saved_fisher import fisher_approx
import stat_functions
import distributions
import fisher
import curves_methods_comparisons as cmc #for cruves estimation plots, take several minutes to process

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.close('all')
plt.ion()
plt.show()


ftsize=13

## Exemple de courbe de fragilité



ref_alph = ref.theta_MLE[0]
ref_bet = ref.theta_MLE[1]

fig = plt.figure(1, figsize=(5.5,4.3))
fig.clf()
ax = fig.add_subplot(111)
ax.plot(ref.a_tab, ref.curve_ML, 'b')
ax.set_xlim(ref.a_tab.min(), ref.a_tab.max())
ax.set_ylim(0,1)
ax.set_xlabel('IM=a', fontsize=ftsize)
ax.set_ylabel('$P_f(a)$', fontsize=ftsize)
# ax.set_title('Courbe de fragilité sismique')
ax.xaxis.set_ticks([])
ax.xaxis.set_ticklabels([], color='green')
ax.yaxis.set_ticks([0,1/2,1])
ax.yaxis.set_ticklabels([r'$0.0$',r'$0.5$',r'$1.0$'], fontsize=ftsize)


fig = plt.figure(2, figsize=(5.5,4.3))
fig.clf()
ax = fig.add_subplot(111)

ax.plot(ref.a_tab, ref.curve_ML, 'b')
ax.set_xlabel('IM=a', fontsize=ftsize)

ax.plot([ref_alph, ref_alph], [0,1/2], '--', color='green', linewidth=0.8)
ax.plot([0,ref_alph], [1/2,1/2], '--', color='green', linewidth=0.8)
f_derivee = lambda x : (x-ref_alph)/ref_alph/ref_bet/np.sqrt(np.pi)+1/2
x1, x2 = 0, ref.a_tab.max()
ax.plot([x1, x2], [f_derivee(x1), f_derivee(x2)], '--', color='magenta', linewidth=0.8)

fig.text(0.23, 0.67, r'slope: $1/\alpha\beta\pi$', color='magenta', fontsize=ftsize)
ax.xaxis.set_ticks([ref_alph])
ax.xaxis.set_ticklabels([r'$\alpha$'], color='green', fontsize=ftsize)
ax.yaxis.set_ticks([0,1/2,1])
ax.yaxis.set_ticklabels([r'$0.0$',r'$0.5$',r'$1.0$'], fontsize=ftsize)
ax.set_xlim(ref.a_tab.min(), ref.a_tab.max())
ax.set_ylim(0,1)


ax.set_ylabel('$P_f(a)$', fontsize=ftsize)
# ax.set_title('Courbe de fragilité sismique')



## Prior de Jeffrey

num_theta = 200
theta_tab = np.zeros((num_theta,2))
theta_tab[:,0] = np.linspace(10**-5, 10, num=num_theta)
theta_tab[:,1] = np.linspace(10**-1,1/2, num=num_theta)
tmin = theta_tab.min()
tmax = theta_tab.max()
theta_grid1, theta_grid2 = np.meshgrid(theta_tab[:,0], theta_tab[:,1])

II = np.zeros((num_theta,num_theta,2,2))
# JJ = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    th = np.concatenate((alpha*np.ones((num_theta,1)),theta_tab[:,1].reshape(num_theta,1)), axis=1)
    II[i,:] = fisher_approx(th)
    # JJ[i,:] = jeffrey_approx(th)
JJ = II[:,:,0,0]*II[:,:,1,1] - II[:,:,0,1]**2

j_min, j_max = 0, np.max(JJ)
levels = np.linspace(j_min, j_max, 15)

fig = plt.figure(5, figsize=(4, 3))
fig.clf()
ax = fig.add_subplot(111)
cm = ax.contourf(theta_grid1, theta_grid2, JJ.T, cmap='viridis', levels=levels)
ax.set_title(r'Jeffreys prior', fontsize=ftsize)
ax.set_xlim(theta_grid1.min(), 6)
ax.set_ylim(theta_grid2.min(), theta_grid2.max())

fig.colorbar(cm)
ax.set_xlabel(r"$\alpha$", fontsize=ftsize)
ax.set_ylabel(r"$\beta$", fontsize=ftsize)
fig.tight_layout()



## Posterior plots

S_tot, A_tot = get_S_A(path, IM, C, quantile=0, rate=100)

init = A_tot.shape[0]
desired = 50
rate = desired/init*100

S, A = get_S_A(path, IM, C, quantile=0, rate=rate, relative=False, shuffle=False)

num_theta = 50
theta_tab = np.zeros((num_theta,2))
theta_tab[:,0] = np.linspace(0.1, A.max(), num=num_theta)
theta_tab[:,1] = np.linspace(2 /10,1, num=num_theta)
tmin = theta_tab.min()
tmax = theta_tab.max()
theta_grid1, theta_grid2 = np.meshgrid(theta_tab[:,0], theta_tab[:,1])

# post from Jeff
pp = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    for j, beta in enumerate(theta_tab[:,1]) :
        # pp[i,j] = stat_functions.log_post_jeff(np.array([alpha,beta]).reshape(1,2),S,A)
        pp[i,j] = stat_functions.log_post_jeff_adapt(np.array([alpha,beta]).reshape(1,2),S,A, fisher_approx)
# pp = pp - pp.max()
ppe = np.exp(pp+15)


j_min, j_max = 0, np.max(ppe)
levels = np.linspace(j_min, j_max, 15)

fig = plt.figure(6, figsize=(4, 3))
fig.clf()
ax = fig.add_subplot(111)
cm = ax.contourf(theta_grid1, theta_grid2, ppe.T, cmap='viridis', levels=levels)
ax.set_title(r'posterior with Jeffreys prior, $n_{data}$'+'={}'.format(A.shape[0]))
ax.set_xlim(theta_grid1.min(), theta_grid1.max())
ax.set_ylim(theta_grid2.min(), theta_grid2.max())

fig.colorbar(cm)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
fig.tight_layout()



#post from GM:
m = 0
a = 0.01
b = 0.01
lamb = 0.01
pp = np.zeros((num_theta,num_theta))

for i,alpha in enumerate(theta_tab[:,0]) :
    for j, beta in enumerate(theta_tab[:,1]) :
        # pp[i,j] = stat_functions.log_post_jeff(np.array([alpha,beta]).reshape(1,2),S,A)
        # pp[i,j] = stat_functions.log_post_jeff_adapt(np.array([alpha,beta]).reshape(1,2),S,A, fisher_approx)
        pp[i,j] = np.log(distributions.gamma_normal_density(np.log(alpha), beta, m, a, b, lamb)) + stat_functions.log_vrais(S,A, np.array([[alpha, beta]]))
# pp = pp - pp.max()
ppe = np.exp(pp+15)


j_min, j_max = 0, np.max(ppe)
levels = np.linspace(j_min, j_max, 15)

fig = plt.figure(7, figsize=(4, 3))
fig.clf()
ax = fig.add_subplot(111)
cm = ax.contourf(theta_grid1, theta_grid2, ppe.T, cmap='viridis', levels=levels)
ax.set_title(r'posterior with G-N prior, $n_{data}$'+'={}'.format(A.shape[0]))
ax.set_xlim(theta_grid1.min(), theta_grid1.max())
ax.set_ylim(theta_grid2.min(), theta_grid2.max())

fig.colorbar(cm)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
fig.tight_layout()



## logDM = f(logIM)

raw_data = get_data(path)
DM = raw_data[['z_max']].values[:500]
PGA = raw_data[['PGA']].values[:500]
SA = raw_data[['sa_5hz']].values[:500]

fig = plt.figure(3, figsize=(4.9,4))
fig.clf()
ax = fig.add_subplot(111)

reg = LinearRegression().fit(np.log(PGA), np.log(DM))

#ax.plot(np.log(PGA), np.log(DM), '.')
ax.loglog(PGA, DM, '.')
(xmin, xmax) = ax.xaxis.get_view_interval()
(ymin, ymax) = ax.yaxis.get_view_interval()
# ax.loglog([xmin,xmax], [10**(reg.coef_*xmin+reg.intercept_/np.log(10))[0], 10**(reg.coef_*xmax+reg.intercept_/np.log(10))[0]], 'r', label='L reg.')
ax.plot([xmin,xmax], [10**(reg.coef_*np.log10(xmin)+reg.intercept_/np.log(10))[0], 10**(reg.coef_*np.log10(xmax)+reg.intercept_/np.log(10))[0]], 'r', label='L reg.')
# ax.xscale('log')
# ax.yscale('log')
ax.set_xlabel('PGA', fontsize=ftsize)
ax.set_ylabel('max. displacement', fontsize=ftsize)
fig.text(0.8, 0.01, r'($m/s^2$)', fontsize=ftsize-3)
fig.text(0, 0.85, r'($m$)', fontsize=ftsize-3, rotation=90)
ax.legend()




fig = plt.figure(4, figsize=(4.9,4))
fig.clf()
ax = fig.add_subplot(111)

reg = LinearRegression().fit(np.log(SA), np.log(DM))

# ax.plot(np.log(SA), np.log(DM), '.')
ax.loglog(SA, DM, '.')
(xmin, xmax) = ax.xaxis.get_view_interval()
(ymin, ymax) = ax.yaxis.get_view_interval()
# ax.plot([xmin,xmax], [(reg.coef_*xmin+reg.intercept_)[0], (reg.coef_*xmax+reg.intercept_)[0]], 'r', label='L reg.')
ax.plot([xmin,xmax], [10**(reg.coef_*np.log10(xmin)+reg.intercept_/np.log(10))[0], 10**(reg.coef_*np.log10(xmax)+reg.intercept_/np.log(10))[0]], 'r', label='L reg.')
ax.set_xlabel('Spectral acceleration 5hz', fontsize=ftsize)
ax.set_ylabel('max. displacement', fontsize=ftsize)
fig.text(0.8, 0.01, r'($m/s^2$)', fontsize=ftsize-3)
fig.text(0, 0.85, r'($m$)', fontsize=ftsize-3, rotation=90)
ax.legend()



## density of A

fig = plt.figure(8, figsize = (4.5,4))
fig.clf()
ax = fig.add_subplot(111)

ax.plot(ref.a_tab, fisher.f_A_mult(ref.a_tab, len(ref.a_tab)), label=r'approx. $f_A$')
ax.hist(fisher.A, 100, density=True, label=r'A hist.')

ax.legend()
ax.set_xlabel(r'$a$=PGA')
ax.set_ylabel(r'$f_A(a)$')
# ax.set_title('')




## convergences comparison

f = open(path+'/for_figs/errors', 'rb')
(err_conf_jeff_tab, err_med_jeff_tab, err_conf_gam_tab, err_med_gam_tab, err_conf_MLE_tab, err_med_MLE_tab) = pickle.load(f)
f.close()

num_est_tab = 20*np.arange(1,11)

fig = plt.figure(9, figsize=(4.5,4))
fig.clf()

ax = fig.add_subplot(111)
ax.plot(num_est_tab, err_conf_MLE_tab, label=r'MLE')
ax.plot(num_est_tab, err_conf_jeff_tab, label=r'Jeff')
# ax.plot(err_conf_gam_tab, label=r'Gamma')
ax.set_title(r'Confidence interval: $|q_{r/2}-q_{1-r/2}|_2$, '+r'$r={}$'.format(cmc.conf), fontsize=ftsize)
ax.legend()
ax.set_xlabel(r'Number of data', fontsize=ftsize)
ax.set_ylabel(r'error', fontsize=ftsize)


fig = plt.figure(10, figsize=(4.5,4))
fig.clf()

ax = fig.add_subplot(111)
ax.plot(num_est_tab, err_med_MLE_tab, label=r'MLE')
ax.plot(num_est_tab, err_med_jeff_tab, label=r'Jeff')
# ax.plot(err_med_gam_tab, label=r'Gamma')
ax.set_title(r'Median error $|q_{med} - P_{ref}|_2$', fontsize=ftsize)
ax.legend()
ax.set_xlabel(r'Number of data', fontsize=ftsize)
ax.set_ylabel(r'error', fontsize=ftsize)



## curves estimations



# ## cruves jeff


fig1 = plt.figure(11, figsize=(4.5,4))
fig1.clf()
ax = fig1.add_subplot(111)

for i,nnn in enumerate(cmc.tab_nn[:-1]) :
    num = cmc.num_est_tab[nnn]
    nn = i
    ax.plot(ref.a_tab, cmc.curve_q1_jeff_tab[nn], label=r'{}'.format(num), color=cmc.colors[i])#, alpha=0.7)
    ax.fill_between(ref.a_tab, cmc.curve_q1_jeff_tab[nn], cmc.curve_q2_jeff_tab[nn], facecolor=cmc.colors[i])


ax.plot(ref.a_tab, ref.curve_ML, color='magenta', label=r'ref')

ax.legend()
ax.set_title(r'Fragility curves convergence (Jeff)', fontsize=ftsize)
ax.set_xlabel(r'a='+IM, fontsize=ftsize)
ax.set_ylabel(r'$P_f(a)$', fontsize=ftsize)

# ## scatter jeff

fig2 = plt.figure(12, figsize=(4.5,4))
fig2.clf()
ax2 = fig2.add_subplot(111)


for i,nnn in enumerate(cmc.tab_nn[:-1]) :
    num = cmc.num_est_tab[nnn]
    nn = i
    ax2.plot(cmc.th_post_jeff_tab[nn,:,0], cmc.th_post_jeff_tab[nn,:,1], 'o', color=cmc.colors[i], label=r'{}'.format(num), markersize=3)

ax2.set_title(r'Posterior simulations from Jeff', fontsize=ftsize)
ax2.set_xlabel(r'$\alpha$', fontsize=ftsize)
ax2.set_ylabel(r'$\beta$', fontsize=ftsize)
ax2.legend()

# ## scatter MLE

fig3 = plt.figure(13, figsize=(4.5,4))
fig3.clf()
ax3 = fig3.add_subplot(111)

for i,nnn in enumerate(cmc.tab_nn[:-1]) :
    num = cmc.num_est_tab[nnn]
    nn = i
    ax3.plot(cmc.th_MLE_tab[nn,:,0], cmc.th_MLE_tab[nn,:,1], 'o', color=cmc.colors[i], label=r'{}'.format(num), markersize=3)

ax3.set_title(r'MLE calculations from bootstrap', fontsize=ftsize)
ax3.set_xlabel(r'$\alpha$', fontsize=ftsize)
ax3.set_ylabel(r'$\beta$', fontsize=ftsize)
ax3.legend()

# ## curves MLE

fig4 = plt.figure(14, figsize=(4.5,4))
fig4.clf()
ax4 = fig4.add_subplot(111)

for i,nnn in enumerate(cmc.tab_nn[:-1]) :
    num = cmc.num_est_tab[nnn]
    nn = i
    ax4.plot(ref.a_tab, cmc.curve_q1_MLE_tab[nn], label=r'{}'.format(num), color=cmc.colors[i])
    ax4.fill_between(ref.a_tab, cmc.curve_q1_MLE_tab[nn], cmc.curve_q2_MLE_tab[nn], facecolor=cmc.colors[i])

ax4.plot(ref.a_tab, ref.curve_ML, color='magenta', label=r'ref')

ax4.legend()
ax4.set_title(r'Fragility curves convergence (MLE)', fontsize=ftsize)
ax4.set_xlabel(r'a='+IM, fontsize=ftsize)
ax4.set_ylabel(r'$P_f(a)$', fontsize=ftsize)



# ### curves gamma

fig5 = plt.figure(15, figsize=(4.5,4))
fig5.clf()
ax5 = fig5.add_subplot(111)

for i,nnn in enumerate(cmc.tab_nn[:-1]) :
    num = cmc.num_est_tab[nnn]
    nn = i
    ax5.plot(ref.a_tab, cmc.curve_q1_gam_tab[nn], label=r'{}'.format(num), color=cmc.colors[i])
    ax5.fill_between(ref.a_tab, cmc.curve_q1_gam_tab[nn], cmc.curve_q2_gam_tab[nn], facecolor=cmc.colors[i])

ax5.plot(ref.a_tab, ref.curve_ML, color='magenta', label=r'ref')

ax5.legend()
ax5.set_title(r'Fragility curves convergence (G-N)', fontsize=ftsize)
ax5.set_xlabel(r'a='+IM, fontsize=ftsize)
ax5.set_ylabel(r'$P_f(a)$', fontsize=ftsize)


# ### scatter gamma

fig6 = plt.figure(16, figsize=(4.5,4))
fig6.clf()
ax6 = fig6.add_subplot(111)

for i,nnn in enumerate(cmc.tab_nn[:-1]) :
    num = cmc.num_est_tab[nnn]
    nn = i
    ax6.plot(cmc.th_post_gam_tab[nn,:,0], cmc.th_post_gam_tab[nn,:,1], 'o', color=cmc.colors[i], label=r'{}'.format(num), markersize=3)

ax6.set_title(r'Posterior simulations from G-N', fontsize=ftsize)
ax6.set_xlabel(r'$\alpha$', fontsize=ftsize)
ax6.set_ylabel(r'$\beta$', fontsize=ftsize)
ax6.legend()


# ## 3 medians

fig1 = plt.figure(17, figsize=(4.5,4))
fig2 = plt.figure(18, figsize=(4.5,4))
fig3 = plt.figure(19, figsize=(4.5,4))
fig1.clf()
fig2.clf()
fig3.clf()

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax3 = fig3.add_subplot(111)

ax1.set_title(r'MLE', fontsize=ftsize)
ax2.set_title(r'Jeffreys prior', fontsize=ftsize)
ax3.set_title(r'Gamma-normal prior', fontsize=ftsize)

for i,nnn in enumerate(cmc.tab_nn) :
    num = cmc.num_est_tab[nnn]
    nn = i
    ax1.plot(ref.a_tab, cmc.curve_med_MLE_tab[nn], label=r'{}'.format(num), color=cmc.colors[i])
    ax2.plot(ref.a_tab, cmc.curve_med_jeff_tab[nn], label=r'{}'.format(num), color=cmc.colors[i])
    ax3.plot(ref.a_tab, cmc.curve_med_gam_tab[nn], label=r'{}'.format(num), color=cmc.colors[i])


ax1.plot(ref.a_tab, ref.curve_ML, color='magenta', label=r'ref')
ax2.plot(ref.a_tab, ref.curve_ML, color='magenta', label=r'ref')
ax3.plot(ref.a_tab, ref.curve_ML, color='magenta', label=r'ref')

ax1.legend()
ax2.legend()
ax3.legend()

ax1.set_xlabel(r'$a=$'+IM, fontsize=ftsize)
ax1.set_ylabel(r'$P_f(a)$', fontsize=ftsize)
ax2.set_xlabel(r'$a=$'+IM, fontsize=ftsize)
ax2.set_ylabel(r'$P_f(a)$', fontsize=ftsize)
ax3.set_xlabel(r'$a=$'+IM, fontsize=ftsize)
ax3.set_ylabel(r'$P_f(a)$', fontsize=ftsize)




### MCMC curves

#1 histogrammes avec moyennes

# fig52, axes52 = plt.subplots(2,2) #histograms 2x2
fig1 = plt.figure(20, figsize=(5.5,5.5))
fig1.clf()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(21, figsize=(5.5,5.5))
fig2.clf()
ax2 = fig2.add_subplot(111)

num_hist = 40
arr_alp_hist, h_alp_hist = np.linspace(2,8, num=num_hist, retstep=True)
arr_bet_hist, h_bet_hist = np.linspace(0.2,1.5, num=num_hist, retstep=True)

for i,nnn in enumerate(cmc.tab_nn) :
    num = cmc.num_est_tab[nnn]
    nn = i

    ax1.hist(cmc.th_post_jeff_tab[nn,:,0], bins=arr_alp_hist+i*h_alp_hist/cmc.num_plots, rwidth=1/cmc.num_plots, label=r'{}'.format(num), color=cmc.colors[i], density=True)
    ax2.hist(cmc.th_post_jeff_tab[nn,:,1], bins=arr_bet_hist+i*h_bet_hist/cmc.num_plots, rwidth=1/cmc.num_plots, label=r'{}'.format(num), color=cmc.colors[i], density=True)
    # axes52[1,1].hist(th_post_gam_tab[nn,:,1], bins=arr_bet_hist+i*h_bet_hist/num_plots, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i], density=True)


ax1.legend()
ax2.legend()
# for ax in axes52.flatten():
#     ax.legend()

# for ax in axes52[0,:] :
ax1.set_xlim(2,6)
ax1.set_xlabel(r'$\alpha$', fontsize=ftsize)
ax1.set_ylabel(r'post$(\alpha)$', fontsize=ftsize)
# for ax in axes52[1,:] :
ax2.set_xlim(0.23,1)
ax2.set_xlabel(r'$\beta$', fontsize=ftsize)
ax2.set_ylabel(r'post$(\beta)$', fontsize=ftsize)


cl_colors = ['cyan', 'gold', 'lightgreen', 'salmon', 'violet', 'lightgrey', 'lightyellow']

axes52bis = []

for axx in [ax1,ax2] :
    axx.set_frame_on(False)
    (xmin, xmax) = axx.xaxis.get_view_interval()
    (ymin, ymax) = axx.yaxis.get_view_interval()
    axx.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax),
                                color = 'black', linewidth = 1.5))
    axx.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin),
                                color = 'black', linewidth = 1.5))

    box = axx.get_position()
    axbis = axx.figure.add_axes(np.concatenate((np.array([box.bounds[0], box.bounds[1]]), box.size)))
    axbis.xaxis.set_label_position('top')
    axbis.yaxis.set_label_position('right')

    axbis.set_frame_on(False)
    axbis.yaxis.tick_right()
    axbis.xaxis.tick_top()
    axbis.xaxis.set_tick_params(color = 'blue', labelcolor = 'blue')
    axbis.yaxis.set_tick_params(color = 'blue', labelcolor = 'blue')

    # axbis.set_ylabel(r'$E[\beta]$  $E[\alpha]$', color='blue')
    axbis.set_xlabel(r'iterations', color='blue', fontsize=ftsize)

    axes52bis.append(axbis)

    # ax22.set_title(r'estimation with posterior')


num_kept_HM_fin = cmc.th_post_jeff_tot_tab[nn,:,0].shape[0]
for i,nnn in enumerate(cmc.tab_nn) :
    num = cmc.num_est_tab[nnn]
    nn = i

    axes52bis[0].plot(cmc.th_post_jeff_tot_tab[nn,:,0].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))
    # axes52bis[1].plot(th_post_gam_tot_tab[nn,:,0].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))
    axes52bis[1].plot(cmc.th_post_jeff_tot_tab[nn,:,1].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))
    # axes52bis[3].plot(th_post_gam_tot_tab[nn,:,1].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))

for j, axbis in enumerate(axes52bis) :
    # (ymin, ymax) = axbis.yaxis.get_view_interval()
    # axbis.set_ylim(ymin,ymax)
    (xmin, xmax) = axbis.xaxis.get_view_interval()
    (ymin, ymax) = axbis.yaxis.get_view_interval()
    axbis.add_artist(plt.Line2D((xmax, xmax), (ymin, ymax),
                              color = 'blue', linewidth = 1.5))
    axbis.add_artist(plt.Line2D((xmin, xmax), (ymax, ymax),
                              color = 'blue', linewidth = 1.5))

    if j==0 :
        axbis.set_ylabel(r'$E[\alpha]$', color='blue', fontsize=ftsize)
    else :
        axbis.set_ylabel(r'$E[\beta]$', color='blue', fontsize=ftsize)


    # axbis.legend()

axes52bis[0].set_title(r'convergence of $\alpha$ from Jeffreys prior', fontsize=ftsize)
axes52bis[1].set_title(r'convergence of $\beta$ from Jeffreys prior', fontsize=ftsize)



fig1 = plt.figure(24, figsize=(5.5,5.5))
fig1.clf()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(25, figsize=(5.5,5.5))
fig2.clf()
ax2 = fig2.add_subplot(111)

num_hist = 40
arr_alp_hist, h_alp_hist = np.linspace(2,8, num=num_hist, retstep=True)
arr_bet_hist, h_bet_hist = np.linspace(0.2,1.5, num=num_hist, retstep=True)

for i,nnn in enumerate(cmc.tab_nn) :
    num = cmc.num_est_tab[nnn]
    nn = i

    ax1.hist(cmc.th_post_gam_tab[nn,:,0], bins=arr_alp_hist+i*h_alp_hist/cmc.num_plots, rwidth=1/cmc.num_plots, label=r'{}'.format(num), color=cmc.colors[i], density=True)
    # axes52[0,1].hist(th_post_gam_tab[nn,:,0], bins=arr_alp_hist+i*h_alp_hist/num_plots, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i], density=True)
    ax2.hist(cmc.th_post_gam_tab[nn,:,1], bins=arr_bet_hist+i*h_bet_hist/cmc.num_plots, rwidth=1/cmc.num_plots, label=r'{}'.format(num), color=cmc.colors[i], density=True)
    # axes52[1,1].hist(th_post_gam_tab[nn,:,1], bins=arr_bet_hist+i*h_bet_hist/num_plots, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i], density=True)


ax1.legend()
ax2.legend()
# for ax in axes52.flatten():
#     ax.legend()

# for ax in axes52[0,:] :
ax1.set_xlim(2,6)
ax1.set_xlabel(r'$\alpha$', fontsize=ftsize)
ax1.set_ylabel(r'post$(\alpha)$', fontsize=ftsize)
# for ax in axes52[1,:] :
ax2.set_xlim(0.23,1)
ax2.set_xlabel(r'$\beta$', fontsize=ftsize)
ax2.set_ylabel(r'post$(\beta)$', fontsize=ftsize)


cl_colors = ['cyan', 'gold', 'lightgreen', 'salmon', 'violet', 'lightgrey', 'lightyellow']

axes52bis = []

for axx in [ax1,ax2] :
    axx.set_frame_on(False)
    (xmin, xmax) = axx.xaxis.get_view_interval()
    (ymin, ymax) = axx.yaxis.get_view_interval()
    axx.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax),
                                color = 'black', linewidth = 1.5))
    axx.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin),
                                color = 'black', linewidth = 1.5))

    box = axx.get_position()
    axbis = axx.figure.add_axes(np.concatenate((np.array([box.bounds[0], box.bounds[1]]), box.size)))
    axbis.xaxis.set_label_position('top')
    axbis.yaxis.set_label_position('right')

    axbis.set_frame_on(False)
    axbis.yaxis.tick_right()
    axbis.xaxis.tick_top()
    axbis.xaxis.set_tick_params(color = 'blue', labelcolor = 'blue')
    axbis.yaxis.set_tick_params(color = 'blue', labelcolor = 'blue')

    # axbis.set_ylabel(r'$E[\beta]$  $E[\alpha]$', color='blue')
    axbis.set_xlabel(r'iterations', color='blue', fontsize=ftsize)

    axes52bis.append(axbis)

    # ax22.set_title(r'estimation with posterior')


num_kept_HM_fin = cmc.th_post_gam_tot_tab[nn,:,0].shape[0]
for i,nnn in enumerate(cmc.tab_nn) :
    num = cmc.num_est_tab[nnn]
    nn = i

    axes52bis[0].plot(cmc.th_post_gam_tot_tab[nn,:,0].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))
    # axes52bis[1].plot(th_post_gam_tot_tab[nn,:,0].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))
    axes52bis[1].plot(cmc.th_post_gam_tot_tab[nn,:,1].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))
    # axes52bis[3].plot(th_post_gam_tot_tab[nn,:,1].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))

for j, axbis in enumerate(axes52bis) :
    # (ymin, ymax) = axbis.yaxis.get_view_interval()
    # axbis.set_ylim(ymin,ymax)
    (xmin, xmax) = axbis.xaxis.get_view_interval()
    (ymin, ymax) = axbis.yaxis.get_view_interval()
    axbis.add_artist(plt.Line2D((xmax, xmax), (ymin, ymax),
                              color = 'blue', linewidth = 1.5))
    axbis.add_artist(plt.Line2D((xmin, xmax), (ymax, ymax),
                              color = 'blue', linewidth = 1.5))

    if j==0 :
        axbis.set_ylabel(r'$E[\alpha]$', color='blue', fontsize=ftsize)
    else :
        axbis.set_ylabel(r'$E[\beta]$', color='blue', fontsize=ftsize)


    # axbis.legend()

axes52bis[0].set_title(r'convergence of $\alpha$ from G-N prior', fontsize=ftsize)
axes52bis[1].set_title(r'convergence of $\beta$ from G-N prior', fontsize=ftsize)



# taux d'acceptations:

fig = plt.figure(22, figsize=(4.5,4))
fig.clf()
ax = fig.add_subplot(111)

for i,nnn in enumerate(cmc.tab_nn) :
    num = cmc.num_est_tab[nnn]
    nn = i

    id_kp = int(cmc.iter_HM/2000)
    ax.plot(id_kp*np.arange(2000), (cmc.accept_jeff_tab[nn].cumsum()/np.arange(1,cmc.iter_HM+1))[id_kp*np.arange(2000)], color=cmc.colors[i], label=r'{}'.format(num), alpha=1)
    # axes53[1].plot(id_kp*np.arange(2000), (accept_gam_tab[nn].cumsum()/np.arange(1,iter_HM+1))[id_kp*np.arange(2000)], color=colors[i], label=r'{}'.format(num), linewidth=0.5, alpha=1)

ax.legend()
ax.set_xlabel(r'iterations', fontsize=ftsize)
ax.set_ylabel(r'avg. accept ratio', fontsize=ftsize)
ax.set_ylim(0,1)

ax.set_title(r'accept rates for Jeffreys posterior', fontsize=ftsize)





fig = plt.figure(23, figsize=(4.5,4))
fig.clf()
ax = fig.add_subplot(111)

for i,nnn in enumerate(cmc.tab_nn) :
    num = cmc.num_est_tab[nnn]
    nn = i

    id_kp = int(cmc.iter_HM/2000)
    ax.plot(id_kp*np.arange(2000), (cmc.accept_gam_tab[nn].cumsum()/np.arange(1,cmc.iter_HM+1))[id_kp*np.arange(2000)], color=cmc.colors[i], label=r'{}'.format(num), alpha=1)
    # axes53[1].plot(id_kp*np.arange(2000), (accept_gam_tab[nn].cumsum()/np.arange(1,iter_HM+1))[id_kp*np.arange(2000)], color=colors[i], label=r'{}'.format(num), linewidth=0.5, alpha=1)

ax.legend()
ax.set_xlabel(r'iterations', fontsize=ftsize)
ax.set_ylabel(r'avg. accept ratio', fontsize=ftsize)
ax.set_ylim(0,1)

ax.set_title(r'accept rates for G-N posterior', fontsize=ftsize)


## histogram only

fig1 = plt.figure(24, figsize=(4.5,4))
fig1.clf()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(25, figsize=(4.5,4))
fig2.clf()
ax2 = fig2.add_subplot(111)

num_hist = 40
arr_alp_hist, h_alp_hist = np.linspace(2,8, num=num_hist, retstep=True)
arr_bet_hist, h_bet_hist = np.linspace(0.2,1.5, num=num_hist, retstep=True)

for i,nnn in enumerate(cmc.tab_nn) :
    num = cmc.num_est_tab[nnn]
    nn = i

    ax1.hist(cmc.th_post_jeff_tab[nn,:,0], bins=arr_alp_hist+i*h_alp_hist/cmc.num_plots, rwidth=1/cmc.num_plots, label=r'{}'.format(num), color=cmc.colors[i], density=True)
    ax2.hist(cmc.th_post_jeff_tab[nn,:,1], bins=arr_bet_hist+i*h_bet_hist/cmc.num_plots, rwidth=1/cmc.num_plots, label=r'{}'.format(num), color=cmc.colors[i], density=True)
    # axes52[1,1].hist(th_post_gam_tab[nn,:,1], bins=arr_bet_hist+i*h_bet_hist/num_plots, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i], density=True)


ax1.legend()
ax2.legend()
# for ax in axes52.flatten():
#     ax.legend()

# for ax in axes52[0,:] :
ax1.set_xlim(2,6)
ax1.set_xlabel(r'$\alpha$', fontsize=ftsize)
ax1.set_ylabel(r'post$(\alpha)$', fontsize=ftsize)
# for ax in axes52[1,:] :
ax2.set_xlim(0.23,1)
ax2.set_xlabel(r'$\beta$', fontsize=ftsize)
ax2.set_ylabel(r'post$(\beta)$', fontsize=ftsize)

ax1.set_title(r'post($\alpha$) histogram from Jeffreys prior', fontsize=ftsize)
ax2.set_title(r'post($\beta$) histogram from Jeffreys prior', fontsize=ftsize)



fig1 = plt.figure(26, figsize=(4.5,4))
fig1.clf()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(27, figsize=(4.5,4))
fig2.clf()
ax2 = fig2.add_subplot(111)

num_hist = 40
arr_alp_hist, h_alp_hist = np.linspace(2,8, num=num_hist, retstep=True)
arr_bet_hist, h_bet_hist = np.linspace(0.2,1.5, num=num_hist, retstep=True)

for i,nnn in enumerate(cmc.tab_nn) :
    num = cmc.num_est_tab[nnn]
    nn = i

    ax1.hist(cmc.th_post_gam_tab[nn,:,0], bins=arr_alp_hist+i*h_alp_hist/cmc.num_plots, rwidth=1/cmc.num_plots, label=r'{}'.format(num), color=cmc.colors[i], density=True)
    # axes52[0,1].hist(th_post_gam_tab[nn,:,0], bins=arr_alp_hist+i*h_alp_hist/num_plots, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i], density=True)
    ax2.hist(cmc.th_post_gam_tab[nn,:,1], bins=arr_bet_hist+i*h_bet_hist/cmc.num_plots, rwidth=1/cmc.num_plots, label=r'{}'.format(num), color=cmc.colors[i], density=True)
    # axes52[1,1].hist(th_post_gam_tab[nn,:,1], bins=arr_bet_hist+i*h_bet_hist/num_plots, rwidth=1/num_plots, label=r'{}'.format(num), color=colors[i], density=True)


ax1.legend()
ax2.legend()
# for ax in axes52.flatten():
#     ax.legend()

# for ax in axes52[0,:] :
ax1.set_xlim(2,6)
ax1.set_xlabel(r'$\alpha$', fontsize=ftsize)
ax1.set_ylabel(r'post$(\alpha)$', fontsize=ftsize)
# for ax in axes52[1,:] :
ax2.set_xlim(0.23,1)
ax2.set_xlabel(r'$\beta$', fontsize=ftsize)
ax2.set_ylabel(r'post$(\beta)$', fontsize=ftsize)

ax1.set_title(r'post($\alpha$) histogram from G-N prior', fontsize=ftsize)
ax2.set_title(r'post($\beta$) histogram from G-N prior', fontsize=ftsize)


### avg alph/bet only


fig1 = plt.figure(28, figsize=(4.7,4))
fig1.clf()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(29, figsize=(4.7,4))
fig2.clf()
ax2 = fig2.add_subplot(111)



ax1.set_xlabel(r'iterations', fontsize=ftsize)
ax2.set_xlabel(r'iterations', fontsize=ftsize)



num_kept_HM_fin = cmc.th_post_jeff_tot_tab[nn,:,0].shape[0]
for i,nnn in enumerate(cmc.tab_nn) :
    num = cmc.num_est_tab[nnn]
    nn = i

    ax1.plot(cmc.th_post_jeff_tot_tab[nn,:,0].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cmc.colors[i], label=r'{}'.format(num))
    # axes52bis[1].plot(th_post_gam_tot_tab[nn,:,0].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))
    ax2.plot(cmc.th_post_jeff_tot_tab[nn,:,1].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cmc.colors[i], label=r'{}'.format(num))
    # axes52bis[3].plot(th_post_gam_tot_tab[nn,:,1].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))



ax1.set_ylabel(r'$E[\alpha]$', fontsize=ftsize)
ax2.set_ylabel(r'$E[\beta]$', fontsize=ftsize)

ax1.legend()
ax2.legend()

ax1.set_title(r'convergence of $\alpha$ from Jeff prior', fontsize=ftsize)
ax2.set_title(r'convergence of $\beta$ from Jeff prior', fontsize=ftsize)






fig1 = plt.figure(30, figsize=(4.7,4))
fig1.clf()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(31, figsize=(4.7,4))
fig2.clf()
ax2 = fig2.add_subplot(111)



ax1.set_xlabel(r'iterations', fontsize=ftsize)
ax2.set_xlabel(r'iterations', fontsize=ftsize)



num_kept_HM_fin = cmc.th_post_gam_tot_tab[nn,:,0].shape[0]
for i,nnn in enumerate(cmc.tab_nn) :
    num = cmc.num_est_tab[nnn]
    nn = i

    ax1.plot(cmc.th_post_gam_tot_tab[nn,:,0].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cmc.colors[i], label=r'{}'.format(num))
    # axes52bis[1].plot(th_post_gam_tot_tab[nn,:,0].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))
    ax2.plot(cmc.th_post_gam_tot_tab[nn,:,1].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cmc.colors[i], label=r'{}'.format(num))
    # axes52bis[3].plot(th_post_gam_tot_tab[nn,:,1].cumsum()/np.arange(1,num_kept_HM_fin+1), '--', color=cl_colors[i], label=r'{}'.format(num))



ax1.set_ylabel(r'$E[\alpha]$', fontsize=ftsize)
ax2.set_ylabel(r'$E[\beta]$', fontsize=ftsize)

ax1.legend()
ax2.legend()

ax1.set_title(r'convergence of $\alpha$ from G-N prior', fontsize=ftsize)
ax2.set_title(r'convergence of $\beta$ from G-N prior', fontsize=ftsize)
























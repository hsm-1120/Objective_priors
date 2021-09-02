# Figures d'illustrations

import os
import inspect
directory = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])) # get script's path
os.chdir(directory)

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from config import path, IM, C
import reference_curves as ref
from data import get_data, get_S_A
from extract_saved_fisher import fisher_approx
import stat_functions
import distributions

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
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

fig.text(0.25, 0.67, r'pente: $1/\alpha\beta\pi$', color='magenta', fontsize=ftsize)
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







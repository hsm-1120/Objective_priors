import os
import inspect
directory = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
os.chdir(directory)

import numpy as np
from matplotlib import pyplot as plt
import scipy.special as spc
import scipy.stats as stat
import math
import pickle
from numba import jit
from scipy import optimize

from data import get_S_A
import reference_curves as ref
import stat_functions
from config import IM, C, path
from utils import rep0, rep1
from extract_saved_fisher import fisher_approx, jeffrey_approx


from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.ion()
plt.show()




## fragility curve estimation via posterior estimation

S_tot, A_tot = get_S_A(path, IM, C, shuffle=False, relative=False)
num_tot = A_tot.shape[0]

num_est = 50
S_ref, A_ref = S_tot[:num_tot], A_tot[:num_tot]


# tirage de nombreux MLE

def func_log_vr(z, a) :
    def log_vr(theta) :
        # return -stat_functions.log_vrais(z, a, theta.reshape(1,2))
        return ref.opp_log_vraiss(theta, a, z)
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

import time
log_post = func_log_post(S_ref, A_ref)
sig_prop = np.array([[0.25,0],[0,0.02]])
a1 = time.time()
th_post, accept = stat_functions.HM_k(t0, log_post, kmax, pi_log=True, max_iter=20, sigma_prop=sig_prop) #long time to execute ; prefer opening files from data bellow
a2 = time.time()

plt.figure()
plt.plot(th_MLE[:,0], th_MLE[:,1], 'x', label='MLE')
plt.plot(th_post[:,0], th_post[:,1], 'o', label='Post')
plt.title(r'Tirages de {} MLE / obs. a posteriori'.format(kmax))
plt.legend()

# file = open(path+r"/th_post", mode='wb')   # To save the results of the Metropolis-Hasting execution
# pickle.dump(th_post, file)
# file.close()
# file = open(path+r"/accept_ratio", mode='wb')
# pickle.dump(accept, file)
# file.close()
#
# file = open(path+r'/th_post', 'rb')   # To load the last results of the Metropolis-Hasting execution instead of executing it again
# th_post = pickle.load(file)           # instead of executing it again
# file.close()
# file = open(path+r'/accept_ratio', 'rb')
# accept = pickle.load(file)
# file.close()



## tracé des quantiles de confiance comparaison avec Ref curve

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


fig = plt.figure(figsize=(18,6))

ax1 = fig.add_subplot(1,2,1)
ax1.plot(ref.a_tab, curve_med_MLE, '-k', label=r'Median')
ax1.plot(ref.a_tab, curve_q1_MLE, '--r', label=r'conf {}%'.format(int((1-conf)*100)))
ax1.plot(ref.a_tab, curve_q2_MLE, '--r')
ax1.plot(A_ref, S_ref, 'og', markersize = 0.5, alpha=0.6)
ax1.plot(ref.a_tab, ref.curve_ML, color='magenta', label=r'ref', alpha=0.7)
ax1.set_title(r'MLE estimation')
ax1.set_ylabel(r'$P_f(a)$')
ax1.set_xlabel(r'a='+IM)
ax1.legend()



ax2 = fig.add_subplot(1,2,2)
ax2.plot(ref.a_tab, curve_med_post, '-k', label=r'Median', linewidth=0.8)
ax2.plot(ref.a_tab, ref.curve_ML, color='magenta', label=r'ref', alpha=0.7, linewidth=0.8)
ax2.plot(ref.a_tab, curve_q1_post, '--r', label=r'conf {}%'.format(int((1-conf)*100)), linewidth=0.8)
ax2.plot(ref.a_tab, curve_q2_post, '--r', linewidth=0.8)
ax2.plot(A_ref, S_ref, 'og', markersize=0.5, alpha=0.6)

ax2.set_frame_on(False)
(xmin, xmax) = ax2.xaxis.get_view_interval()
(ymin, ymax) = ax2.yaxis.get_view_interval()
ax2.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax),
                              color = 'black', linewidth = 1.5))
ax2.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin),
                              color = 'black', linewidth = 1.5))
ax2.set_ylabel(r'$P_f(a)$')
ax2.set_xlabel(r'a='+IM)
ax2.legend()



box = ax2.get_position()
ax22 = fig.add_axes(np.concatenate((np.array([box.bounds[0], box.bounds[1]]), box.size)))
# ax22 = fig.add_subplot(1,2,2)
# ax2 = plt.gca().twinx()
ax22.xaxis.set_label_position('top')
ax22.yaxis.set_label_position('right')

ax22.plot(accept.mean(axis=1), '--b', label=r'accpet ratio', linewidth=0.5, alpha=0.3)

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
ax22.add_artist(plt.Line2D((xmax, xmax), (ymin, ymax),
                              color = 'blue', linewidth = 1.5))
ax22.add_artist(plt.Line2D((xmin, xmax), (ymax, ymax),
                              color = 'blue', linewidth = 1.5))
ax22.set_ylabel(r'accept ratio', color='blue')
ax22.set_xlabel(r'iterations', color='blue')
ax22.legend()


ax22.set_title(r'estimation with posterior')




















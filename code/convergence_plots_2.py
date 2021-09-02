
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numba import jit
import numpy.random as rd

from config import IM, C, path
from data import get_S_A
import stat_functions
import reference_curves as ref
from utils import rep0, rep1
from extract_saved_fisher import fisher_approx, jeffrey_approx
from distributions import log_gamma_normal_pdf

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.ion()
plt.show()


## data

S_tot, A_tot = get_S_A(path, IM, C, shuffle=False, relative=False)
num_tot = A_tot.shape[0]



sigma_prop = np.sqrt(0.4*np.array([[0.5,0],[0,0.2]]))

num_est_max = 200
num_est_min = 10
number_estimations = 19
# num_est_tab = np.arange(num_est_min, num_est_max)
num_est_tab = 10*np.arange(1,20)

iter_HM = 20000
keep_HM_each = 500
nb_diff_tirages = 100
nb_tot_HM = keep_HM_each*nb_diff_tirages

nb_MLE = 2000



th_MLE_tab = np.zeros((number_estimations, nb_MLE, 2))
th_post_jeff_tab = np.zeros((number_estimations, nb_tot_HM, 2))
th_post_gam_tab = np.zeros((number_estimations, nb_tot_HM, 2))
accept_jeff_tab = np.zeros((number_estimations, iter_HM))
accept_gam_tab = np.zeros((number_estimations, iter_HM))

th_post_jeff_tot_tab = np.zeros((number_estimations, iter_HM, nb_diff_tirages, 2))
th_post_gam_tot_tab = np.zeros((number_estimations, iter_HM, nb_diff_tirages, 2))


t0 = np.array([3,0.3])


### Tirages de MLE

def func_log_vr(z, a) :
    def log_vr(theta) :
        return -stat_functions.log_vrais(z, a, theta.reshape(1,2))
        # return ref.opp_log_vraiss(theta, a, z)
    return log_vr

print('start MLE')
for nn, num in enumerate(num_est_tab) :
    # # #todo : implement a true bootstrap for MLE  #maybe starting from a fake one as the other data will also beneficiate from an fake bootstrap
    # calculate kmax th_MLE via bootstrap:
    for k in range(nb_MLE) :
        i = rd.randint(0, num_tot-num_est_max)
        S = S_tot[i:i+num]+0
        A = A_tot[i:i+num]+0
        # print(S)
        # if len(S)==0 :
        #     print(i,num,k)
        #     break
        log_vr = func_log_vr(S,A)
        th_MLE_tab[nn, k] = optimize.minimize(log_vr, t0, bounds=[(0.01,100),(0.01,100)], options={'maxiter':10, 'disp':False}).x


    # if nn%10==0 :
    print(r'{}/{}'.format(nn, number_estimations))



### Tirages de th_post_jeff

def func_log_post(num) :
    ids = np.arange(num_tot)
    rd.shuffle(ids)
    z, a = S_tot[ids], A_tot[ids]
    @jit(nopython=True)
    def log_post(theta) :
        s = theta.shape[0]
        pp = np.zeros((s,1))
        for i in range(s) :
            th = np.zeros((1,2))
            th[0] = theta[i] +0
            pp[i] = stat_functions.log_post_jeff_adapt(th,z[i:i+num],a[i:i+num], Fisher=fisher_approx)
        return pp[:,0]
    return log_post







print('starting Jeff post')

for nn, num in enumerate(num_est_tab) :


    #simulate kmax_conv th_post_jeffrey via HM
    log_post = func_log_post(num) #todo : change it to bootstrap
    sigma_prop = np.array([[0.1,0],[0,0.095]])
    # t_fin, t_tot, acc = stat_functions.adaptative_HM_k(t0, log_post, num_sim_HM, pi_log=True, max_iter=iter_HM, sigma0=sigma_prop)
    t_fin, t_tot, acc = stat_functions.adaptative_HM_k(t0, log_post, nb_diff_tirages, pi_log=True, max_iter=iter_HM) #attention: de base HM_k c'est pas ca mais ca revient surement au meme: a voir
    th_post_jeff_tab[nn, :, 0] = t_tot[-keep_HM_each:,:, 0].flatten()
    th_post_jeff_tab[nn, :, 1] = t_tot[-keep_HM_each:,:, 1].flatten()
    # accept_jeff_tab[nn] = np.minimum(acc,1).mean(axis=1)
    accept_jeff_tab[nn] = np.minimum(acc,1).mean(axis=1)
    # th_post_jeff_tot_tab[nn] = t_tot[:,0]+0
    th_post_jeff_tot_tab[nn,:,:,0] = t_tot[:,:,0]
    th_post_jeff_tot_tab[nn,:,:,1] = t_tot[:,:,1]

    # if nn%10==0 :
    print(r'{}/{}'.format(nn, number_estimations))



# print('step 1 done' )



### ? Tirages de th_post_Gam

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


print('Starting Gamma post')

for nn, num in enumerate(num_est_tab) :
    #todo : copy paste above to sectgion bellow
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






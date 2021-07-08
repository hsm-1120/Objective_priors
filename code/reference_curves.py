import numpy as np
import pylab as plt
import math
from scipy.special import erf
from scipy import optimize
from data import get_S_A
from config import C, IM, path
from utils import rep0, rep1


##

S_tot, A_tot = get_S_A(path, IM, C, shuffle=False)
n = A_tot.shape[0]

num_a = 100
a_tab, h_a = np.linspace(10**-5, 10, num=num_a, retstep=True)

F_tab = np.zeros(num_a)
for i,a in enumerate(a_tab[:-1]) :
    A_in_id = (A_tot>=a)*(A_tot<a_tab[i+1])
    fm = S_tot[A_in_id]
    if len(fm)==0 :
        F_tab[i] = 0
    else :
        F_tab[i] = fm.mean()
F_tab[-1] = 1


# n = len(F_tab)
# sigma = np.sqrt(F_tab.var())
# h = sigma*n**(-1/5)
F_tab_lis = F_tab * np.exp(-(rep0(a_tab,num_a)-rep1(a_tab,num_a))**2/(2*h_a**2)).sum(axis=0)

# F_tab_lis = np.zeros(num_a)
# for i,a in enumerate(a_tab[:-1]) :
#     F_tab_lis[i] = (np.exp(-(A_tot-a)**2/(2*h_a)**2)*S_tot).mean()
F_tab_lis = F_tab_lis / F_tab_lis.max()


def opp_log_vraiss(theta) :
    a = A_tot+0
    z = S_tot+0
    # l = theta.shape[0]
    n = a.shape[0]
    logp = np.zeros((n,1))
    # ind0 = np.zeros(l, dtype='int')
    #for i,t in enumerate(theta) :
    if np.any(theta<=0) :
        return np.inf
    else:
        # for k,zk in enumerate(z) :
        for k in range(n) :
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



theta_MLE = optimize.minimize(opp_log_vraiss, np.array([3,0.3])).x
curve_ML = 1/2 + 1/2 * erf(np.log(a_tab/theta_MLE[0])/theta_MLE[1])


if __name__=="__main__":
    plt.ion()
    plt.show()

    plt.figure()
    plt.plot(a_tab, F_tab, label='MC')
    plt.plot(a_tab, F_tab_lis, label='Kernel')
    plt.plot(a_tab, curve_ML, label=r'MLE, $\theta=({:4.2f},{:4.2f})$'.format(theta_MLE[0], theta_MLE[1]))
    plt.plot(A_tot, S_tot, 'x')
    plt.xlim((a_tab.min(), a_tab.max()))
    plt.xlabel('a='+IM)
    plt.ylabel(r'$P_f(a)$')
    plt.title(r'Reference fragility curves, n_data={}'.format(A_tot.shape[0]))
    plt.legend()
















import numpy as np
import numpy.random as rd
from numba import jit
from utils import simpson_numb


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
def HM_k(z0, pi, k, max_iter=5000, sigma_prop=10**-2) :
  d = z0.shape[0]
  z_v = jrep0(z0, k)
  for n in range(max_iter) :
    for i,z_vi in enumerate(z_v) :
      zi = z_vi + sigma_prop@np.eye(d)@rd.randn(d)
      pi_z_vi = pi(z_vi)
      pi_zi = pi(zi)
      log_alpha = np.log(pi_z_vi)-np.log(pi_zi)
      rand = np.log(rd.rand())<log_alpha
      z_v[i] += rand*(zi-z_vi)
  return z_v


def HM_Fisher() :
    return True




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




import pickle
from numba import jit, prange
import numpy as np
import save_fisher
from config import path, save_fisher_arr

##

file = open(path+r'Fisher_array', 'rb')
I = pickle.load(file)
file.close()

J = I[:,:,0,0]*I[:,:,1,1] - I[:,:,0,1]**2


@jit(nopython=True, parallel=True, cache=True) #todo : verify if parallel do not cause any error here
def fisher_approx(theta_array, cut=True) :
    l = theta_array.shape[0]
    Fis = np.zeros((l,2,2))
    almin = save_fisher_arr.alpha_min
    almax = save_fisher_arr.alpha_max
    bemin = save_fisher_arr.beta_min
    bemax = save_fisher_arr.beta_max
    tmax = np.array([almax, bemax])
    tmin = np.array([almin, bemin])
    # for k, theta in enumerate(theta_array) :
    for k in prange(l) :
        theta = theta_array[k]
        if np.any(theta>tmax) or np.any(theta<tmin) :
            Fis[k] = 0
        else :
            i = np.argmin(np.abs(save_fisher.theta_tab1-theta[0]))
            j = np.argmin(np.abs(save_fisher.theta_tab2-theta[1]))
            Fis[k] = I[i,j]+0
    return Fis


@jit(nopython=True)
def jeffrey_approx(tehta_array, cut=True) :
    jeff = np.zeros((theta_array.shape[0]))
    almin = save_fisher_arr.alpha_min
    almax = save_fisher_arr.alpha_max
    bemin = save_fisher_arr.beta_min
    bemax = save_fisher_arr.beta_max
    tmax = np.array([almax, bemax])
    tmin = np.array([almin, bemin])
    for k, theta in enumerate(theta_array) :
        if np.any(theta>tmax) or np.any(theta<tmin) :
            jeff[k] = 0
        else :
            i = np.argmin(np.abs(save_fisher.theta_tab1-theta[0]))
            j = np.argmin(np.abs(save_fisher.theta_tab2-theta[1]))
            jeff[k] = J[i,j]+0
    return jeff











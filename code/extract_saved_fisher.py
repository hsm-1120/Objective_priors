
import pickle
from numba import jit, prange
import numpy as np
import pylab as plt
from config import path, save_fisher_arr

##

file = open(path+r'/Fisher_array', 'rb')
I = pickle.load(file)
file.close()

J = I[:,:,0,0]*I[:,:,1,1] - I[:,:,0,1]**2

almin = save_fisher_arr.alpha_min
almax = save_fisher_arr.alpha_max
bemin = save_fisher_arr.beta_min
bemax = save_fisher_arr.beta_max
theta_tab1 = save_fisher_arr.alpha_tab
theta_tab2 = save_fisher_arr.beta_tab

@jit(nopython=True, parallel=True) #verify if parallel do not cause any error here
def fisher_approx(theta_array, cut=True) :
    l = theta_array.shape[0]
    Fis = np.zeros((l,2,2))
    tmax = np.array([almax, bemax])
    tmin = np.array([almin, bemin])
    # for k, theta in enumerate(theta_array) :
    for k in prange(l) :
        theta = theta_array[k]
        if np.any(theta>tmax) or np.any(theta<tmin) :
            Fis[k] = 0
        else :
            i = np.argmin(np.abs(theta_tab1-theta[0]))
            j = np.argmin(np.abs(theta_tab2-theta[1]))
            Fis[k] = I[i,j]+0
    return Fis


@jit(nopython=True, parallel=True)
def jeffrey_approx(theta_array, cut=True) :
    l = theta_array.shape[0]
    jeff = np.zeros(l)
    tmax = np.array([almax, bemax])
    tmin = np.array([almin, bemin])
    # for k, theta in enumerate(theta_array) :
    for k in prange(l) :
        theta = theta_array[k]
        if np.any(theta>tmax) or np.any(theta<tmin) :
            jeff[k] = 0
        else :
            i = np.argmin(np.abs(theta_tab1-theta[0]))
            j = np.argmin(np.abs(theta_tab2-theta[1]))
            jeff[k] = J[i,j]+0
    return jeff



if __name__=="__main__":
    plt.ion()
    plt.show()

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

    plt.figure()
    axes = plt.axes(projection="3d")
    axes.plot_surface(theta_grid1, theta_grid2, JJ.T)

    plt.title('Jeffreys maillage fin')
    axes.set_xlabel('alpha')
    axes.set_ylabel('beta')
    axes.set_zlabel('J')








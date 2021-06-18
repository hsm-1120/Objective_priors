import os

os.chdir(r"Z:/code")

import pylab as plt
import numpy as np
import fisher
from config import IM, C, path
from data import get_S_A

plt.ion()
plt.show()

##

S, A = get_S_A(path, IM, C)

num_theta = 3
theta_tab = np.zeros((num_theta,2))
theta_tab[:,0] = np.linspace(0.1, A.max(), num=num_theta)
theta_tab[:,1] = np.linspace(1/10,1/2, num=num_theta)
tmin = theta_tab.min()
tmax = theta_tab.max()
theta_grid1, theta_grid2 = np.meshgrid(theta_tab[:,0], theta_tab[:,1])

a_tab, h_a = np.linspace(10**-10, 2*A.max(), num=5, retstep=True)

import time
atime = time.time()
JJ = fisher.Fisher_Simpson_Numb(theta_tab[:,0], theta_tab[:,1], a_tab)
btime = time.time()
import pylab as plt
import numpy as np
import fisher
from config import IM, C, path
from data import get_S_A

plt.ion()
plt.show()

##

S, A = get_S_A(path, IM, C)

num_theta = 20
theta_tab = np.zeros((num_theta,2))
theta_tab[:,0] = np.linspace(0.1, A.max(), num=num_theta)
theta_tab[:,1] = np.linspace(1/10,1/2, num=num_theta)
tmin = theta_tab.min()
tmax = theta_tab.max()
theta_grid1, theta_grid2 = np.meshgrid(theta_tab[:,0], theta_tab[:,1])

##


J_MC = fisher.Jeffreys_MC(theta_tab[:,0], theta_tab[:,1], A)


a_tab, h_a = np.linspace(10**-10, 2*A.max(), num=1000, retstep=True)
J_t = fisher.Jeffreys_rectangles(theta_tab[:,0], theta_tab[:,1], a_tab, h_a)


a_tab, h_a = np.linspace(10**-10, 2*A.max(), num=100, retstep=True)
J_s = fisher.Jeffreys_simpson(theta_tab[:,0], theta_tab[:,1], a_tab)



plt.figure(1)
plt.clf()
axes = plt.axes(projection="3d")
axes.plot_surface(theta_grid1, theta_grid2, J_MC)

plt.title('Jeffreys Monte-Carlo')
axes.set_xlabel('alpha')
axes.set_ylabel('beta')
axes.set_zlabel('J_MC')



plt.figure(2)
plt.clf()
axes = plt.axes(projection="3d")
axes.plot_surface(theta_grid1, theta_grid2, J_t)

plt.title('Jeffreys approx. rectangles')
axes.set_xlabel('alpha')
axes.set_ylabel('beta')
axes.set_zlabel('J_t')



plt.figure(3)
plt.clf()
axes = plt.axes(projection="3d")
axes.plot_surface(theta_grid1, theta_grid2, J_s)

plt.title('Jeffreys via Simson')
axes.set_xlabel('alpha')
axes.set_ylabel('beta')
axes.set_zlabel('J_s')

##
# test de méthode de simpson via numba

a_tab, h_a = np.linspace(10**-10, 2*A.max(), num=30, retstep=True)

import time
atime = time.time()
JJ = fisher.Fisher_Simpson_Numb(theta_tab[:,0], theta_tab[:,1], a_tab)
btime = time.time()

##

c= time.time()
JJJ = fisher.Jeffreys_simpson(theta_tab[:,0], theta_tab[:,1], a_tab)
d= time.time()




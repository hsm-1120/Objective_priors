#import os
#os.chdir(r"Z:/code")

import numpy as np

path = r'../data'
IM = 'PGA'
C = 0.8*10**-2

class thet_arrays():
    def __init__(self, alpha_min, alpha_max, beta_min, beta_max, num_a, num_b) :
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_alpha = num_a
        self.num_beta = num_b
        al_t, h_a = np.linspace(alpha_min, alpha_max, num_a, retstep=True)
        be_t, h_b = np.linspace(beta_min, beta_max, num_b, retstep=True)
        self.alpha_tab = al_t
        self.h_alpha = h_a
        self.beta_tab = be_t
        self.h_beta = h_b

save_fisher_arr = thet_arrays(10**-5, 10, 10**-3, 2, 500, 500)




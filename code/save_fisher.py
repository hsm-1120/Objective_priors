
import numpy as np
import pickle
from config import path, save_fisher_arr
import fisher


## Calcul et sauvegarde d'un maillage fin de Jeffrey


function = fisher.Fisher_Simpson

alpha_min = save_fisher_arr.alpha_min
alpha_max = save_fisher_arr.alpha_max
beta_min = save_fisher_arr.beta_min
beta_max = save_fisher_arr.beta_max
num = save_fisher_arr.num_alpha

theta_tab1 = save_fisher_arr.alpha_tab
theta_tab2 = save_fisher_arr.beta_tab

# a_tab = np.linspace(10**-5, 10, )

# I = fisher.Fisher_Simpson(theta_tab1, theta_tab2, a_tab)

def save_fisher() :
    I = np.zeros((num,num,2,2))

    for i,alpha in enumerate(theta_tab1) :
        for j, beta in enumerate(theta_tab2) :
            a_tab = np.exp(np.linspace(np.log(alpha)-4*beta, np.log(alpha)+4*beta, 40))
            I[i,j] = function(np.array([alpha]).reshape(1,1), np.array([beta]).reshape(1,1), a_tab)
        if i%10 ==0 :
            print("i={}/{}".format(i,num))
            # print(alpha)
            # break
        # break

    file = open(path+"Fisher_array", mode='wb')
    pickle.dump(I, file)
    file.close()

    return True

if __name__=="__main__":
    save_fisher()








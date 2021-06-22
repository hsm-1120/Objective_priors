import os

os.chdir(r"Z:/code")


## Plot KL(posterior|Jeffreys) as a function of the number of observations

# dim_k = 1000
# range_k = np.arange(1000)*100 # len(S) = 10000
#
# #jit
# def iterate_KL(range_k, dim_k) :
#   KL = np.zeros(dim_k)
#   for i,k in enumerate(range_k) :
#     z = S[:k,0]
#     a = A[:k,0]
#     J = Jeffreys_func(A[:,0])
#     theta_simul = HM_k(np.ones(2), J, 1000)
#     post_const = np.mean(p_z_cond_a_theta_binary(z,a,theta_simul))
#     KL[i] = Kullback_Leibler(posterior_func(z, a, post_const, J)/post_const, J)
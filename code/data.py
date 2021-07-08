import pandas as pd
import numpy as np

def get_data(path) :
    return pd.read_csv(path+r'/KH_xi=2%_sa.csv')

def get_S_A(path, IM, C_S=10**-2, shuffle=True, quantile=0, rate=100, relative=True) :
    data = get_data(path)
    A = data[[IM]].values
    qA1 = np.quantile(A, quantile)
    qA2 = np.quantile(A, 1-quantile)
    S = 1*(data[['z_max']].values>=C_S)
    # n = len(A)
    # qint = int(quantile*n)
    S = S[(A>qA1)*(A<qA2)]
    A = A[(A>qA1)*(A<qA2)]


    n = len(A)
    if shuffle :
        n = len(A)
        ids = np.arange(n)
        np.random.shuffle(ids)
        A = A[ids]+0
        S = S[ids]+0

    if relative==True :
        n0 = (S==0).sum()
        n1 = (S==1).sum()
        ind0 = np.arange(n)[S==0][:int(n0*rate/100)]
        ind1 = np.arange(n)[S==1][:int(n1*rate/100)]
        ind = np.concatenate((ind0,ind1))
        A = A[ind]+0
        S = S[ind]+0
    else :
        num = int(n*rate/100)
        A = A[:num]+0
        S = S[:num]+0

    if shuffle :
        n = len(A)
        ids2 = np.arange(n)
        np.random.shuffle(ids2)
        A = A[ids2]+0
        S = S[ids2]+0

    return S, A
import pandas as pd
import numpy as np

def get_data(path) :
    return pd.read_csv(path+r'/KH_xi=2%_sa.csv')

def get_S_A(path, IM, C_S=10**-2, shuffle=True) :
    data = get_data(path)
    A = data[[IM]].values
    S = 1*(data[['z_max']].values>=C_S)
    n = len(A)
    ids = np.arange(n)
    np.random.shuffle(ids)
    A = A[ids]+0
    S = S[ids]+0
    return S, A
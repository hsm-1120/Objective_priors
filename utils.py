from numba import jit
import numpy as np




## array manipulation usefull functions :

def rep0(array, N) :
    """create an axis 0 on the array, and repeat it N times over axis 0"""
    return array[np.newaxis].repeat(N, axis=0)

def rep1(array, N) :
    """create an axis -1 on the array, and repeat it N times over axis -1"""
    return array[...,np.newaxis].repeat(N, axis=-1)



@jit(nopython=True, cache=True)
def jrep0(array, N) :
    """same than rep0 but in numba mode"""
    art = np.expand_dims(array, axis=0)
    arr = np.expand_dims(array, axis=0)
    for k in range(N-1) :
        arr = np.append(arr, art, axis=0)
    return arr

@jit(nopython=True, cache=True)
def jrep1(array, N) :
    """same than rep1 but in numba mode"""
    art = np.expand_dims(array, axis=-1)
    arr = np.expand_dims(array, axis=-1)
    for k in range(N-1) :
        arr = np.append(arr, art, axis=-1)
    return arr


## for integral calculations

@jit(nopython=True,  cache=True)
def simpson_numb(y, x) :
    n = y.shape[0]
    lim_id = (n+1)//2
    id_05 = (2*np.arange(n)+1)[:lim_id-1]
    id_1 = (2*np.arange(n))[:lim_id]
    y_05 = y[id_05]
    y_1 = y[id_1]
    h = x[1]-x[0]
    integ = h/3 * (y_1[:-1]+y_1[1:])/2 + 2*y_05
    return integ.sum()
























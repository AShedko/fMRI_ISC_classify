import numpy as np
from scipy.stats.stats import pearsonr
from helpers import *

from matplotlib import pyplot as plt
from const import *

def corr_xyz(L, s1,s2, x, y, z):
    """
    Correlation in xyz voxel between s1 and s2
    :param s1: subject 1
    :param s2: subject 2
    :param x: 
    :param y: 
    :param z: 
    :return: 
    """
    X1,X2 = np.ndarray(SIZE,np.float32), np.ndarray(SIZE,np.float32)
    for i in range(1,SIZE):
        n1,n2 = L.get_img(s1,i),L.get_img(s2,i)
        X1[i]=(n1[x, y, z])
        X2[i]=(n2[x, y, z])
        if i%300 ==0: print(i)
    ker = np.ones(CONV_WINDOW)/CONV_WINDOW
    plt.plot(X1,'b', label='X1')
    X1 = np.convolve(X1, ker, 'simple')[40:-21]
    plt.plot(X1, 'r', label='X1_avg')
    plt.plot(X2, 'g', label='X2')
    X2 = np.convolve(X2, ker, 'simple')[40:-21]
    plt.plot(X2, 'm', label='X2_avg')
    print(X1,X2)
    plt.show()
    return pearsonr(X1, X2)


def ISC_mat_voxel(l, x, y, z, target = range(1,SIZE+1)):
    NSUBJ = 29
    """
    Correlation in xyz voxel between all subjects
    :param L: Loader
    :param x: 
    :param y: 
    :param z: 
    :return: 
    """
    X = [np.ndarray(len(target)+1, np.float32) for j in range(NSUBJ+1)]
    for k in range(1,NSUBJ+1):
        i=0
        for t in target:
            X[k][i] = l.get_img(k,t)[x, y, z]
            i+=1
            # if i%300 ==0: print(i)
        print(k)
    ker = np.ones(CONV_WINDOW)/CONV_WINDOW
    X = list(map (lambda x : np.convolve(x, ker, 'valid')[CONV_WINDOW:-CONV_WINDOW], X))
    return np.corrcoef(X)

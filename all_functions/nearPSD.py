import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
import os
import random as r
from scipy.stats import multivariate_normal
from numpy.linalg import inv



def nearPSD(A, epsilon=1e-23):
    n = A.shape[0]
    eigval, eigvec = np.linalg.eig(A)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1 / (np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    out = B * B.T
    return (out)
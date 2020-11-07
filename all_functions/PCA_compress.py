import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
import os
import random as r
from scipy.stats import multivariate_normal
from numpy.linalg import inv



def PCA_compress(dimensionality, normal_data, anomalous_data):
    n_components = dimensionality
    pca_train = decomposition.PCA(n_components=n_components)
    pca_train.fit(normal_data)
    normal_data = pca_train.transform(normal_data)
    anomalous_data = pca_train.transform(anomalous_data)
    return normal_data, anomalous_data
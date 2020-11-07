import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
import os
import random as r
from scipy.stats import multivariate_normal
from numpy.linalg import inv
from all_functions import nearPSD



def plot_gaussians(train_data, H, Mu, Sd, W):
    Cov = np.random.rand(H, train_data.shape[1], train_data.shape[1])
    xx, yy = np.mgrid[-5:5:.1, -5:5:.1]
    pos = np.dstack((xx, yy))
    plt.figure()
    plt.grid()
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axes().set_aspect('equal', 'datalim')
    plt.scatter(train_data[:, 0], train_data[:, 1], c="blue", alpha=0.5)
    plt.xlabel("First PC", fontsize=15)
    plt.ylabel("Second PC", fontsize=15)

    for h in range(H):
        Cov[h] = Sd[h] * np.identity(Cov[h].shape[-2])
        try:
            rv = multivariate_normal(Mu[h], Cov[h])
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.axes().set_aspect('equal', 'datalim')
            plt.contour(xx, yy, W[0][h]*rv.pdf(pos))
            plt.xlabel("First PC", fontsize=15)
            plt.ylabel("Second PC", fontsize=15)

        except ValueError:
            Cov[h] = nearPSD.nearPSD(Cov[h])
            rv = multivariate_normal(Mu[h], Cov[h])
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.axes().set_aspect('equal', 'datalim')
            plt.contour(xx, yy, W[0][h]*rv.pdf(pos))
            plt.xlabel("First PC", fontsize=15)
            plt.ylabel("Second PC", fontsize=15)

    plt.show()
    return Cov
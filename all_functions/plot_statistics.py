import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
import os
import random as r
from scipy.stats import multivariate_normal
from numpy.linalg import inv



def plot_statistics(Avg_Error, Avg_output, Avg_l2_weights, Avg_sum_variances):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(Avg_Error, 'r')
    plt.grid()
    plt.ylabel("Avg. Cost per Epoch")
    plt.xlabel("Epochs")
    plt.subplot(2, 2, 2)
    plt.plot(Avg_output)
    plt.grid()
    plt.ylabel("Avg. Output per Epoch")
    plt.xlabel("Epochs")
    plt.subplot(2, 2, 3)
    plt.plot(Avg_l2_weights)
    plt.grid()
    plt.ylabel("Avg. L-2 of Weights per Epoch")
    plt.xlabel("Epochs")
    plt.subplot(2, 2, 4)
    plt.grid()
    plt.plot(Avg_sum_variances)
    plt.ylabel("Avg. Sum of Variances per Epoch")
    plt.xlabel("Epochs")
    plt.show()
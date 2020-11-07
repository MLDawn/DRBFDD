import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
import os
import random as r
from scipy.stats import multivariate_normal
from numpy.linalg import inv



def plot_decision_boundary(cov, m, W):
    # Plot the decision boundary
    plt.figure()
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    output = []
    for i in range(xx.shape[1]):
        for j in range(yy.shape[0]):
            a = xx[0][i]
            b = yy[j][0]
            X = np.array([[a, b]])
            invcov = inv(cov)
            a = (X - m[:, np.newaxis])
            # # Optimised
            d = np.matmul(np.matmul(a, invcov), np.transpose(a, [0, 2, 1]))
            #################
            # Now the array P(Hxn) has ALL the likelihoods of ALL the training data of ALL the Kernels
            P = np.exp(-0.50 * np.diagonal(d, axis1=1, axis2=2))
            # Here we have a column-wise multiplication between P(Hxn) and Weights (CAN be BETTER CODED)
            # Z(n) is what is what the output neuron has received
            Z = np.sum((P.T * W).T, axis=0)
            # Here we apply the Lecun's recommended tanh() function to get the outputs
            # The output vector has n number of elements
            # Check whether EBFDD or EBFDD-tanh is requested
            output.append(1.7159 * np.tanh(float(2 / 3) * Z))
    output = np.reshape(np.array(output), (xx.shape[1], yy.shape[0]))
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axes().set_aspect('equal', 'datalim')
    plt.contourf(x, y, output.T)
    plt.colorbar()
    plt.clim(0,1)
    plt.xlabel('First PC', fontsize=15)
    plt.ylabel('Second PC', fontsize=15)
    plt.show()
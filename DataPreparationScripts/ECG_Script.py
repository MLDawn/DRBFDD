import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning) #turn off irrelevant scipy future warning
from DataPreparationScripts import heartbeat as hb
from DataPreparationScripts import normalizer
import importlib
import random
import time
import pandas as pd
import numpy as np
from collections import Counter
from scipy import signal
from scipy.signal import find_peaks, resample
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import listdir
from os.path import isfile, join
import sys
import warnings
from numpy import savetxt
from numpy import loadtxt
importlib.reload(hb)
importlib.reload(normalizer)
from numpy import asarray
from numpy import savetxt


def get_key(val, my_dict):
    """
    Simple Function to Get Key
    in Dictionary from val.

    Input: Key, Dictionary
    Output: Val

    """
    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"

def prepare_ECG(normal, anomalous, path, resample=False):
    classes= {0:'N',1:'L',2:'R',3:'V',4:'/',5:'A',6:'f',7:'F',8:'!',9:'j'}
    normal = int(normal)
    anomalous = int(anomalous)

    print("Top classes to be loaded are:\n")
    print(classes)

    # ##############################################################################################
    # isolation algorithim heart beat data
    # x_signal, x_v, y = hb.isolate_patient_data(patients=hb.all_patients(),classes=classes,
    #                     classes_further=hb.classes_further, classes_reducer=None,
    #                      min_HR=40, max_HR=140, fs=360, verbose=False, plot_figs=False)
    #
    # if resample:
    #     print('Resampling the data from current 360 (Hz) sampling rateto half of it: 180-190 (Hz) ...')
    #     x_signal = hb.resample_vals(x_signal, samp_len=187)
    #     x_v = hb.resample_vals(x_v, samp_len=187)
    #
    # # merge x_signal and x_v along the second dimension (we need to add a second dimension to each one first i.e., #channels)
    # x_signal = np.expand_dims(x_signal, axis=1)
    # x_v = np.expand_dims(x_v, axis=1)
    #
    # x = np.concatenate((x_signal,x_v), axis=1)
    #
    # y = np.array([get_key(y_i, classes) for y_i in y[:, 2]])
    #
    # np.save('x.npy', x)
    # np.save('y.npy', y)


    #########################################################################################################

    x = np.load(path+'x.npy')
    y = np.load(path+'y.npy')


    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # separate normal and the requested anomalous beats
    normal_idx = np.where(y == normal)
    anomalous_idx = np.where(y == anomalous)

    # Grab the relevant data and labels
    normal_data = x[normal_idx]
    anomalous_data = x[anomalous_idx]

    # # Randomly sample 5000 heartbeats
    # np.random.seed(0)
    # rand_idx = np.random.choice(normal_data.shape[0], 30000)
    # normal_data = normal_data[rand_idx]

    # regenerate the labels with the same lengths
    normal_data_label = np.zeros(normal_data.shape[0])
    anomalous_data_label = np.ones(anomalous_data.shape[0])




    return normal_data, normal_data_label, anomalous_data, anomalous_data_label


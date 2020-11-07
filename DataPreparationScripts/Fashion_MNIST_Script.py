import torch
from torchvision import datasets, transforms
import os

import random as r

import numpy as np

import torch

def seeder(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    r.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seeder(1)


trainset = datasets.FashionMNIST('data/FMNIST/', download=True, train=True, transform=None)
testset = datasets.FashionMNIST('data/FMNIST/', download=True, train=False, transform=None)

def normalize(a):
   n = 2. * (a - np.min(a)) / np.ptp(a) - 1
   return n

def TRAIN_SIZE():

    x_train = torch.reshape(trainset.data, (trainset.data.shape[0], 784))
    y_train = trainset.targets

    return x_train, y_train


def TEST_SIZE():
    x_test = torch.reshape(testset.data, (testset.data.shape[0], 784))
    y_test = testset.targets

    return x_test, y_test


def prepare_FMNIST(normal_digits, anomalous_digits):
    All_training_digits, All_training_digits_labels = TRAIN_SIZE()
    All_testing_digits, All_testing_digits_labels = TEST_SIZE()

    All_training_digits = All_training_digits.data.numpy()
    All_training_digits_labels = All_training_digits_labels.data.numpy()

    All_testing_digits = All_testing_digits.data.numpy()
    All_testing_digits_labels = All_testing_digits_labels.data.numpy()

    normal_data = []
    normal_data_label = []
    anomalous_data = []
    anomalous_data_label = []
    # We would like all that is normal, inside All_training_digits to be extracted for train_data no label needed(RBFDD: Unsupervised)
    # We would like all that is anomalous, inside All_training_digits to be extracted for test_data and test_label
    for i in range(All_training_digits.shape[0]):
        label = All_training_digits_labels[i]
        if label in normal_digits:
            normal_data.append(All_training_digits[i])
            normal_data_label.append(0)
        elif label in anomalous_digits:
            anomalous_data.append(All_training_digits[i])
            anomalous_data_label.append(1)
    for i in range(All_testing_digits.shape[0]):
        label = All_testing_digits_labels[i]
        if label in normal_digits:
            normal_data.append(All_testing_digits[i])
            normal_data_label.append(0)
        elif label in anomalous_digits:
            anomalous_data.append(All_testing_digits[i])
            anomalous_data_label.append(1)
    # take the whole All_testing_images as the test_data that is used for stradified sampling and prediction by RBFDD.test()

    normal_data = np.array(normal_data, dtype=np.float32)/255.0
    normal_data_label = np.array(normal_data_label)


    anomalous_data = np.array(anomalous_data, dtype=np.float32)/255.0
    anomalous_data_label = np.array(anomalous_data_label)

    return normal_data, normal_data_label, anomalous_data, anomalous_data_label














# import numpy as np
# import pandas as pd
#
# def prepare_Fashion_MNIST(normal, anomalous):
#     dataset = pd.read_csv('D:\PhD\Benchmark Datasets and Papers\Fashionmnist\Fashion-mnist_train.csv')
#     X = dataset[dataset.columns[1:]]
#     Y = np.array(dataset["label"])
#     # Now let's put them in the desired shape
#     normal_data = []
#     normal_data_label = []
#     anomalous_data = []
#     anomalous_data_label = []
#     for i in range(len(X)):
#         if Y[i] in normal:
#             normal_data.append(X.iloc[i].values)
#             normal_data_label.append(Y[i])
#         elif Y[i] in anomalous:
#             anomalous_data.append(X.iloc[i].values)
#             anomalous_data_label.append(Y[i])
#     # Now about the test data
#     dataset = pd.read_csv('D:\PhD\Benchmark Datasets and Papers\Fashionmnist\Fashion-mnist_test.csv')
#     X = dataset[dataset.columns[1:]]
#     Y = np.array(dataset["label"])
#
#     for i in range(len(X)):
#         if Y[i] in normal:
#             normal_data.append(X.iloc[i].values)
#             normal_data_label.append(Y[i])
#         elif Y[i] in anomalous:
#             anomalous_data.append(X.iloc[i].values)
#             anomalous_data_label.append(Y[i])
#
#     normal_data = np.array(normal_data, dtype=np.float64)
#     normal_data_label = np.array(normal_data_label, dtype=np.float64)
#     anomalous_data = np.array(anomalous_data, dtype=np.float64)
#     anomalous_data_label = np.array(anomalous_data_label, dtype=np.float64)
#
#     # Normalize
#     normal_data = normal_data/255
#     anomalous_data = anomalous_data/255
#
#     return normal_data, normal_data_label, anomalous_data, anomalous_data_label
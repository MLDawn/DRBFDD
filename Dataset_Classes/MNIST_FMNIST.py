import random as r
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from all_functions import PCA_compress

os.environ['PYTHONHASHSEED'] = str(1)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)  # if you are using multi-GPU.
np.random.seed(1)  # Numpy module.
r.seed(1)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Datasets(Dataset):
    def __init__(self, dataset, normal_label, anomalous_label, compress = False):
        self.normal = normal_label
        self.anomalous = anomalous_label
        self.normal_data = None
        self.normal_data_label = None


        self.anomalous_data = None
        self.anomalous_data_label = None
        self.boot_strap_train_data = None
        self.boot_strap_train_index = None
        self.concatenated_test_data = None
        self.concatenated_test_label = None
        self.mode = None
        self.input_dim = None
        # get all the data

        if dataset == 'MNIST':
            from DataPreparationScripts import MNIST_Script as MNIST_Handler
            # get all the data
            [self.normal_data, self.normal_data_label, self.anomalous_data, self.anomalous_data_label] = MNIST_Handler.prepare_MNIST(
                                                                                                                 self.normal,
                                                                                                                 self.anomalous)

            # for the sake of AEN, record the input dimension, which helps with choosing the encode_dim hyper parameter for the AEN
            self.input_dim = self.normal_data.shape[1]


        elif dataset == 'FMNIST':
            from DataPreparationScripts import Fashion_MNIST_Script as MNIST_Handler
            # get all the data
            [self.normal_data, self.normal_data_label, self.anomalous_data,
             self.anomalous_data_label] = MNIST_Handler.prepare_FMNIST(
                self.normal,
                self.anomalous)
            # for the sake of AEN, record the input dimension, which helps with choosing the encode_dim hyper parameter for the AEN
            self.input_dim = self.normal_data.shape[1]
            print("===============================================================")



        # If you need to compress your data, this bit will be executed
        if compress == True:
            self.normal_data, self.anomalous_data = PCA_compress.PCA_compress(2, self.normal_data, self.anomalous_data)
            self.input_dim = self.normal_data.shape[1]

    def __len__(self):
        if self.mode == 'train':
            return self.boot_strap_train_data.shape[0]
        elif self.mode == 'eval':
            return self.concatenated_test_data.shape[0]


    def __getitem__(self, index):
        sample = dict()
        if self.mode == 'train':
            sample = {'data': self.boot_strap_train_data[index]}
        elif self.mode == 'eval':
            sample = {'data': self.concatenated_test_data[index], 'targets': self.concatenated_test_label[index]}

        return sample

    '''
    This prepares the training set by randomly sampling sample_size% of the windowed self.normal_data and self.normal_data_label
    '''
    def prepare_training_data(self, sample_size):
        # bootstrap sample from the train_data
        boot_strap_size = int(self.normal_data.shape[0] * sample_size)
        # Generate a uniform random sample from np.arange(boot_strap_size) of size boot_strap_size:
        # generate the index of the sampled values NO REPLACEMENT
        self.boot_strap_train_index = np.random.choice(self.normal_data.shape[0], boot_strap_size, replace=False)
        # extract the associated boot_strap_train data for the RBFDD training (No need for labels)
        self.boot_strap_train_data = self.normal_data[self.boot_strap_train_index]


    '''
    This prepares the testing set
    '''
    def prepare_testing_data(self):
        test_data_normal_portion = np.delete(self.normal_data, self.boot_strap_train_index, axis=0)
        test_label_normal_portion = np.delete(self.normal_data_label, self.boot_strap_train_index, axis=0)
        # Now concatenate these newly extracted normal samples to the existing
        # anomalous data to create the the final test set for the prediction
        concatenated_test_data = np.concatenate((test_data_normal_portion, self.anomalous_data), axis=0)
        concatenated_test_label = np.concatenate((test_label_normal_portion, self.anomalous_data_label), axis=0)
        self.concatenated_test_data, self.concatenated_test_label = concatenated_test_data, concatenated_test_label
        # Before passing the test data, let's shuffle it to avoid ANY STREAMIMG assumption
        # self.concatenated_test_data, self.concatenated_test_label = shuffle(concatenated_test_data, concatenated_test_label,
        #                                                           random_state=0)
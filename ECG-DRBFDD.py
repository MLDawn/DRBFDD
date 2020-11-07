import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import yaml
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import itertools as it
import time
import sys
from all_algorithms.DeepRBFDD import DeepRBFDD
from all_functions import  seed, k_means, popoviciu
import torch.distributed as dist
from Dataset_Classes.ECG import ECG
import os

#==============================================================================
seed.seed(42)
try:
    DEVICE_index = str(sys.argv[1])
except:
    DEVICE_index = str(0)
# os.environ["CUDA_VISIBLE_DEVICES"]=DEVICE_index
IDs = [int(i) for i in DEVICE_index.split(',')]
device = torch.device("cuda:"+DEVICE_index if torch.cuda.is_available() else "cpu")


with open('parameters.yaml', 'r') as f:
    parameters = yaml.safe_load(f)
DATA = parameters['Dataset']
normal = DATA['normal']
anomalous = DATA['anomalous']
dataset = DATA['name']
path = DATA['path']



shared = parameters['Shared']
MAX_EPOCH = shared['max_epoch']
ETA = shared['eta']
batch_size = shared['batch_size']
SAMPLING_TIMES = shared['sampling_times']
sample_size = shared['sample_size']
RBFDD_params = parameters['RBFDD']
BETA = RBFDD_params['beta']
LAMBDA = RBFDD_params['lamda']
algorithm = RBFDD_params['algorithm']
rbfdd_hidden_fraction = RBFDD_params['hidden_fraction']

DeepNetOptions = parameters['DeepNetOptions']


deep_name = 'CNN1D'
CNN1D_params = parameters['CNN1D']
pre_rbfdd_input_dim = CNN1D_params['pre_rbfdd_input_dim']
use_kmeans = CNN1D_params['use_kmeans']
mini_batch_kmeans = CNN1D_params['mini_batch_kmeans']



def train(model, dataset, batch_size, num_epochs, eta, beta, Lambda):

    print(deep_name + " is running ...")

    div = False
    dataset.mode = 'train'
    if torch.cuda.device_count() > 1 and len(str(DEVICE_index).split(','))>1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        print('Device IDs are' + str(IDs))
        model = nn.DataParallel(model)

    model.to(device)


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)


    pre_rbfdd = torch.zeros((1, pre_rbfdd_input_dim)).to(device)
    # Switch on the train mode in the model
    model.eval()
    with torch.no_grad():
        for D in dataloader:
            data = D['data']
            # Add channel of size 3 for each imag
            if len(data.shape) < 3:
                data = data.reshape(data.shape[0], 1, data.shape[1])

            data = data.to(device)
            # Forward partial propagation
            # cc = model.module.partial_forward(data)
            try:
                pre_rbfdd = torch.cat((pre_rbfdd, model.module.partial_forward(data)), 0)
            except:
                pre_rbfdd = torch.cat((pre_rbfdd, model.partial_forward(data)), 0)


    # Drop the first column of zeros
    pre_rbfdd = pre_rbfdd.narrow(0, 1, pre_rbfdd.shape[0] - 1)
    pre_rbfdd = pre_rbfdd.detach().cpu().numpy()




    if use_kmeans:
        try:
            [mu, sd, div] = k_means.k_means(pre_rbfdd, model.module.H, device, mini_batch_kmeans)
        except:
            [mu, sd, div] = k_means.k_means(pre_rbfdd, model.H, device, mini_batch_kmeans)
    else:
        minimum = np.min(pre_rbfdd)
        maximum = np.max(pre_rbfdd)
        [mu, sd] = popoviciu.popoviciu(pre_rbfdd.shape[1], h, minimum=minimum, maximum=maximum)
        print("Centers and Spread are Initialized by Popoviciu...")

    if div == False:
        try:
            model.module.rbfdd.Mu.data.copy_(mu.data)
            model.module.rbfdd.Sd.data.copy_(sd.data)
        except:
            model.rbfdd.Mu.data.copy_(mu.data)
            model.rbfdd.Sd.data.copy_(sd.data)



        optimizer = torch.optim.Adam(model.parameters(), lr=eta)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        model.train()
        for epoch in range(num_epochs):
            start = time.time()

            cum_E = []

            for D in dataloader:
                # Clear the gradients
                optimizer.zero_grad()
                data = D['data']

                if len(data.shape) < 3:
                    data = data.reshape(data.shape[0], 1, data.shape[1])

                data = data.to(device)

                y_hat = model(data)

                if len(y_hat.size()) != 0:

                    try:
                        w = model.module.rbfdd.fc.weight
                        sd = model.module.rbfdd.Sd
                    except:
                        w = model.rbfdd.fc.weight
                        sd = model.rbfdd.Sd
                    reg_w = torch.sum(w ** 2)
                    reg_sd = torch.sum(sd ** 2)
                    E = torch.sum(0.50 * (((1 - y_hat) ** 2) + beta * reg_sd + Lambda * reg_w))
                    if np.isnan(E.data.cpu().detach().numpy()):
                        print("NaN values found in the cost function ...")
                        print("Divergent Parameters Found ...")
                        div = True
                        return model, div
                    cum_E.append(E.data)
                    E.backward()
                    # Compute the regularized terms of the error function
                    optimizer.step()


                else:
                    div = True
                    print("Divergent Parameters Found ...")
                    return model, div

            end = time.time()

            if (epoch + 1) % 10 == 0:
                print('Epoch %d --- SE= %.5f --- %.2f(s)/epoch' % (epoch+1,sum(cum_E), (end-start)))

    return model, div

def evaluation(model, dataset):

    # Set the self.mode in the dataset to train
    dataset.mode = 'eval'
    # Switch on the evaluation mode in the model
    model.eval()
    # Build the dataloader

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True)
    ground_truth = []
    output = []

    with torch.no_grad():
        for D in dataloader:
            data = D['data']
            if len(data.shape) < 3:
                data = data.reshape(data.shape[0], 1, data.shape[1])
            ground_truth.append(D['targets'].detach().cpu().numpy())
            data = data.to(device)
            # Forward propagation
            y_hat = model(data)
            output.append(y_hat.detach().cpu().numpy())


    output = list(chain(*output))
    ground_truth = list(chain(*ground_truth))

    # Cut the little extra bit from output to make sure GT and output are the same size
    # output = output[:len(ground_truth)]


    output = np.array(output)
    ground_truth = np.array(ground_truth)


    output = output.reshape(len(output))
    ground_truth = ground_truth.reshape(len(ground_truth))


    try:
        auc_score = roc_auc_score(ground_truth, -output)
        print("--- AUC = %.4f" % auc_score)
    except:

        pass


    return output, ground_truth

#============================================================================================
hyper_combo = list(it.product(ETA, MAX_EPOCH, BETA, LAMBDA))
# ===========================================================================================

d = ECG(normal, anomalous, path)
H = [int(x * pre_rbfdd_input_dim) for x in rbfdd_hidden_fraction]

# Create the dataset object which upon creation separates normal and anomaoous data
round_counter = 1
total_number_rounds = len(H) * len(hyper_combo)
for h in H:
    # Some of these might end up as 0 so we will skip them
    if h == 0:
        round_counter +=1
        continue
    final_result = dict()
    for combo in hyper_combo:
        bp_eta = combo[0]
        bp_epoch = combo[1]
        beta = combo[2]
        _lambda = combo[3]

        current_hyper_parameters = [h, bp_epoch, bp_eta, beta, _lambda]
        # We give the benefit of doubt to every combination to be not divergent
        divergent = False
        step_result = []

        for sampleing_times in range(SAMPLING_TIMES):


            deeprbfdd = DeepRBFDD(h, pre_rbfdd_input_dim, DeepNetOption=deep_name,device=device)


            print("Current Algorithm: " + deep_name+'-'+algorithm +'-batchsize:%d' % batch_size + '\n'+
                  'Dataset being sampled: MIT-BIH Arrhythmia' + '-(N,A): ' + str((normal,anomalous))+ 'batchsize: '+str(batch_size)+'\n'+
                  'Current Round: %d ' % round_counter + 'Out of: %d' % total_number_rounds + ' Rounds'+'\n'+
                  'Current Hyper-Parameters: (H, BP_epoch, BP_eta, beta(spreads), lambda(weights)): (%d, %d, %.5f, %.5f, %.5f)' %
                  (h, bp_epoch, bp_eta, beta, _lambda)+'\n' + 'Sample Number: %d ' % (sampleing_times + 1) + 'Out of: %d' % SAMPLING_TIMES + ' Sampling Rounds'+'\n')
            d.prepare_training_data(sample_size)
            d.prepare_testing_data()
            # Call the train function
            [deeprbfdd, divergent] = train(deeprbfdd, d, batch_size, bp_epoch, bp_eta, beta, _lambda)

            if divergent == False:
                # Test the learner
                [raw_output, ground_truth] = evaluation(deeprbfdd, d)
                # Save results of this round
                step_result.append(
                    [raw_output, ground_truth])
            else:
                step_result = []
                break
        # Increment the round counter
        round_counter = round_counter + 1


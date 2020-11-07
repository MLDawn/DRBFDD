import torch
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
from Dataset_Classes.MNIST_FMNIST import Datasets as DATASETS
from all_algorithms.DeepRBFDD import DeepRBFDD
from all_functions import seed, k_means, popoviciu



#==============================================================================
seed.seed(42)
use_gpu = torch.cuda.is_available()
print("GPU Available: ", use_gpu)
try:
    DEVICE_index = str(sys.argv[1])
except:
    DEVICE_index = str(0)
device = torch.device("cuda:"+str(DEVICE_index) if use_gpu else "cpu")
#===============================================================================
with open('parameters.yaml', 'r') as f:
    parameters = yaml.safe_load(f)
# From Shared import the parameters shared between the models
DATA = parameters['Dataset']
normal = DATA['normal']
anomalous = DATA['anomalous']
dataset_name = DATA['name']
dataset_collaction = {dataset_name:[[normal, anomalous]]}
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



deep_name = ''
if DeepNetOptions['ResNet'] == True and DeepNetOptions['LeNet'] == False:
    resnet_params = parameters['ResNet']
    resnet_fine_tune = resnet_params['fine_tune']
    resnet_fine_tune_layers = resnet_params['fine_tune_layers_num']
    pre_rbfdd_input_dim = resnet_params['pre_rbfdd_input_dim']
    deep_name = 'ResNet'
    use_kmeans = resnet_params['use_kmeans']

elif DeepNetOptions['ResNet'] == False and DeepNetOptions['LeNet'] == True:
    deep_name = 'LeNet'
    Lenet_params = parameters['LeNet']
    pre_rbfdd_input_dim = Lenet_params['pre_rbfdd_input_dim']
    use_kmeans = Lenet_params['use_kmeans']


def train(model, dataset, batch_size, num_epochs, eta, beta, Lambda):

    print(deep_name + " is running ...")

    div = False
    dataset.mode = 'train'
    model = model.to(device)


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    pre_rbfdd = torch.zeros((1, pre_rbfdd_input_dim)).to(device)

    # Switch on the train mode in the model
    model.eval()
    with torch.no_grad():
        for D in dataloader:
            data = D['data']
            # Add channel of size 3 for each image
            if deep_name == 'ResNet' or deep_name == 'LeNet':
                data = data.resize_(batch_size, 1, 28, 28)

            data = data.to(device)
            # Forward partial propagation
            # cc = model.module.partial_forward(data)
            pre_rbfdd = torch.cat((pre_rbfdd, model.partial_forward(data)), 0)

    # Drop the first column of zeros
    # pre_rbfdd = pre_rbfdd.transpose(dim0=0, dim1=1)
    pre_rbfdd = pre_rbfdd.narrow(0, 1, pre_rbfdd.shape[0] - 1)
    pre_rbfdd = pre_rbfdd.detach().cpu().numpy()
    # pre_rbfdd = np.transpose(pre_rbfdd)


    if use_kmeans:
        [mu, sd, div] = k_means.k_means(pre_rbfdd, model.H, device)
    else:
        minimum = np.min(pre_rbfdd)
        maximum = np.max(pre_rbfdd)
        [mu, sd] = popoviciu.popoviciu(pre_rbfdd.shape[1], h, minimum=minimum, maximum=maximum)
        print("Centers and Spread are Initialized by Popoviciu...")

    if div == False:
        model.rbfdd.Mu.data.copy_(mu.data)
        model.rbfdd.Sd.data.copy_(sd.data)

        # print('Current Round: %d ' % round_counter + 'Out of: %d' % total_number_rounds + ' Rounds'+'\n')


        optimizer = torch.optim.Adam(model.parameters(), lr=eta)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        model.train()
        for epoch in range(num_epochs):

            cum_E = []

            for D in dataloader:
                # Clear the gradients
                optimizer.zero_grad()
                data = D['data']
                if deep_name == 'ResNet' or deep_name == 'LeNet':
                    data = data.resize_(batch_size, 1, 28, 28)


                data = data.to(device)

                y_hat = model(data)


                if len(y_hat.size()) != 0:

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
                    print("Divergent Parameters Found ...")
                    return model, div

            if (epoch + 1) % 10 == 0:
                print('Epoch %d --- SE= %.5f' % (epoch+1,sum(cum_E)))

    return model, div

def evaluation(model, dataset):

    # Set the self.mode in the dataset to train
    dataset.mode = 'eval'
    # Switch on the evaluation mode in the model
    model.eval()
    # Build the dataloader

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    ground_truth = []
    output = []

    with torch.no_grad():
        for D in dataloader:
            data = D['data']
            ground_truth.append(D['targets'].detach().cpu().numpy())
            if deep_name == 'ResNet' or deep_name == 'LeNet':
                data = data.resize_(batch_size, 1, 28, 28)
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

for dataset in dataset_collaction.keys():
    for scenario in dataset_collaction[dataset]:
        normal = scenario[0]
        anomalous = scenario[1]
        d = DATASETS(dataset, scenario[0], scenario[1], False)
        H = [int(x * pre_rbfdd_input_dim) for x in rbfdd_hidden_fraction]

        # Create the dataset object which upon creation separates normal and anomaoous data
        round_counter = 1
        total_number_rounds = len(H) * len(hyper_combo)
        for h in H:
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

                    if deep_name == 'ResNet':
                        print("Resnet is running ...")
                        deeprbfdd = DeepRBFDD(h, pre_rbfdd_input_dim, device, fine_tune=resnet_fine_tune,
                                              fine_tune_layers=resnet_fine_tune_layers,
                                              DeepNetOption='ResNet')

                    elif deep_name == 'LeNet':
                        print("LeNet is running ...")
                        deeprbfdd = DeepRBFDD(h, pre_rbfdd_input_dim, device,DeepNetOption=deep_name)


                    print("Current Algorithm: " + deep_name+'-'+algorithm +'\n'+
                          'Dataset being sampled: ' + dataset_name + ' Scenario: ' + str(scenario)+'\n'+
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
                            [raw_output, ground_truth, deeprbfdd.rbfdd.Mu, deeprbfdd.rbfdd.Sd,
                             deeprbfdd.rbfdd.fc.weight])
                    else:
                        step_result = []
                        break

                # Increment the round counter
                round_counter = round_counter + 1
                if divergent == False:
                    final_result[str(current_hyper_parameters)] = step_result



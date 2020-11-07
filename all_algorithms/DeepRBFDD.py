import torch
from torch import nn
from torchvision import models
from all_algorithms.RBFDD import RBFDD
from all_algorithms.Lenet import LeNet
from all_algorithms.Resnet import Resnet
from all_algorithms.CNN1D import CNN1D

class DeepRBFDD(nn.Module):
    def __init__(self, h, pre_rbfdd_input_dim, device, fine_tune=False, fine_tune_layers = [],DeepNetOption = 'ResNet'):
        super(DeepRBFDD, self).__init__()
        # Load the pre-trained model
        self.device= device
        self.H = h
        self.pre_rbfdd_input_dim = pre_rbfdd_input_dim
        self.DeepNetOption = DeepNetOption
        self.fine_tune = fine_tune
        self.fine_tune_layers = fine_tune_layers

        assert DeepNetOption == 'ResNet' or DeepNetOption == 'LeNet' or DeepNetOption == 'CNN1D', "Deepname should be either of these"


        if self.DeepNetOption == 'ResNet':
            self.resnet = Resnet(self.fine_tune, self.fine_tune_layers)

        elif self.DeepNetOption == 'LeNet':
            self.lenet = LeNet()

        elif self.DeepNetOption == 'CNN1D':
            self.cnn1d = CNN1D()




        # Load the EBFDD network that would be loaded
        self.rbfdd = RBFDD(h, self.pre_rbfdd_input_dim,self.device)


    def forward(self, x):
        if self.DeepNetOption == 'ResNet':
            x = self.resnet(x)
            x = torch.squeeze(x)

        elif self.DeepNetOption == 'LeNet':
            x = self.lenet(x)

        elif self.DeepNetOption == 'CNN1D':
            x = self.cnn1d(x)


        # APPLY Lecun's non-linearity here
        x = 1.7159 * torch.tanh(float(2 / 3) * x)
        x, _ = self.rbfdd(x)
        return x

    def partial_forward(self, x):
        if self.DeepNetOption == 'ResNet':
            x = self.resnet(x)
            x = torch.squeeze(x)

        elif self.DeepNetOption == 'LeNet':
            x = self.lenet(x)

        elif self.DeepNetOption == 'CNN1D':
            x = self.cnn1d(x)
            x = torch.squeeze(x)


        # APPLY Lecun's non-linearity here
        x = 1.7159 * torch.tanh(float(2 / 3) * x)
        return x
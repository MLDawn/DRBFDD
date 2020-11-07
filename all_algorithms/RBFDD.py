import torch
from torch import nn
import numpy as np




class RBFDD(nn.Module):
    def __init__(self, h, pre_rbfdd_input_dim, device):
        super(RBFDD, self).__init__()

        self.div = False
        self.device = device
        # Number of hidden units (i.e., Gaussian Kernels)
        self.H = h
        self.pre_rbfdd_input_dim = pre_rbfdd_input_dim
        # Define the new parameters as class Parameter
        self.Sd = torch.nn.Parameter(torch.zeros((self.H,)))
        self.Mu = torch.nn.Parameter(torch.zeros((self.H, self.pre_rbfdd_input_dim)))
        # Define the linear layer, so you could have weights beteen the Gaussian layer and the output layer
        self.fc = nn.Linear(self.H, 1, bias=False)

    def Gaussian(self, x):
        p = torch.tensor([])
        try:
            # with torch.no_grad():
            #     if np.isnan(self.Sd.data.cpu().detach().numpy()).any():
            #         temp_sd = self.Sd.data.cpu().detach().numpy()
            #         nan_idx = np.argwhere(np.isnan(temp_sd)).squeeze()
            #         if nan_idx.size < self.H:
            #             self.self_prune(nan_idx)
            #             print("--- %d node(s) have been pruned..." % nan_idx.size)
            #             self.pruned = True

            a = x - self.Mu[:, None]
            b = torch.matmul(a, torch.transpose(a, dim0=1, dim1=2))
            numinator = torch.diagonal(b, dim1=1, dim2=2)
            spread_square = self.Sd ** 2
            denum = spread_square[:, None]
            power = -0.50 * torch.div(numinator, denum)
            power = self.clip_power(power)
            p = torch.exp(power)
            p = p.transpose(1, 0)
        except RuntimeError:
            print('Memory Error ...')
            self.div = True
        return p

    # def self_prune(self, to_prune_idx):
    #     # Update H
    #     self.H = self.H - to_prune_idx.size
    #     # Update Mu
    #
    #     mu_numpy = self.Mu.data.cpu().detach().numpy()
    #     mu_numpy = np.delete(mu_numpy, to_prune_idx,axis=0)
    #     self.Mu= torch.nn.Parameter(torch.tensor(mu_numpy, dtype=torch.float32).to(self.device))
    #
    #     # Update Sd
    #     sd_numpy = self.Sd.data.cpu().detach().numpy()
    #     sd_numpy = np.delete(sd_numpy, to_prune_idx, None)
    #     self.Sd = torch.nn.Parameter(torch.tensor(sd_numpy, dtype=torch.float32).to(self.device))
    #
    #     # Update W
    #     w_numpy = self.fc.weight.cpu().detach().numpy()
    #     w_numpy = np.delete(w_numpy, to_prune_idx, None)
    #     w_numpy = w_numpy.reshape((1, len(w_numpy)))
    #     self.fc = nn.Linear(self.H, 1, bias=False)
    #     self.fc.weight = torch.nn.Parameter(torch.tensor(w_numpy, dtype=torch.float32).to(self.device))



    def clip_power(self, power):
        minimum = torch.tensor(-100.).to(self.device)
        maximum = torch.tensor(40.).to(self.device)
        power = torch.where(power < minimum, minimum, power)
        power = torch.where(power > maximum, maximum, power)

        return power

    def forward(self, x):
        self.div = False # Reset
        y = torch.tensor([])
        p = self.Gaussian(x)
        if self.div is True:
            return y, p

        z = self.fc(p)
        y = 1.7159 * torch.tanh(float(2 / 3) * z)
        return y, p

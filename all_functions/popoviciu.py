import torch
import numpy as np


def popoviciu(pre_rbfdd_dim, H, minimum, maximum):

    bounded_var = ((maximum - minimum)**2)/4
    sd = np.sqrt(np.array(H*[bounded_var]))
    mu = np.random.uniform(low=minimum , high= maximum , size=(H, pre_rbfdd_dim))

    Sd = torch.tensor(sd, dtype=torch.float32)
    Mu = torch.tensor(mu, dtype=torch.float32)

    return Mu, Sd


import torch.nn as nn
import torch.nn.functional as F
import torch
# SOURCE: https://www.marktechpost.com/2019/07/30/introduction-to-image-classification-using-pytorch-to-classify-fashionmnist-dataset/


class LeNet(nn.Module):
    def __init__(self, flag=False):

        super(LeNet, self).__init__()
        if flag == True:
            in_channel = 3
            flattened = 400
        else:
            in_channel=1
            flattened = 256



        self.cnn_model = nn.Sequential(
        nn.Conv2d(in_channel, 6, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(2, stride=2))
        self.fc_model = nn.Sequential(
            nn.Linear(256, 120),
            nn.Tanh(),
            nn.Linear(120, 84))


    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)

        return x
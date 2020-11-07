import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models

# SOURCE: https://www.marktechpost.com/2019/07/30/introduction-to-image-classification-using-pytorch-to-classify-fashionmnist-dataset/


class Resnet(nn.Module):
    def __init__(self, fine_tune, fine_tune_layers, compress=False):

        super(Resnet, self).__init__()
        self.compress = compress
        self.resnet = models.resnet18(pretrained=True)
        # Convert it to single channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.fine_tune = fine_tune
        self.fine_tune_layers = fine_tune_layers
        # Pop the last fully connected layer of resnet
        self.in_features = self.resnet.fc.in_features  # This shows the number of outputs of resnet after amputating it
        # Strip off the last layer
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        if self.compress == True:
            self.fc1 = nn.Linear(self.in_features, int(self.in_features / 2))
            self.fc2 = nn.Linear(int(self.in_features / 2), int(self.in_features / 4))
            self.fc3 = nn.Linear(int(self.in_features / 4), int(self.in_features / 8))
            self.fc4 = nn.Linear(int(self.in_features / 8), int(self.in_features / 16))


        # Make Resnet a fixed feature extractor
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Activate fine-tunning of certain layers if Requested
        if self.fine_tune == True:
            for name, param in self.resnet.named_parameters():
                if (param.requires_grad == False):
                    for layer in self.fine_tune_layers:
                        if name.startswith(str(layer)):
                            param.requires_grad = True


    def forward(self, x):
        x = self.resnet(x)
        x = torch.squeeze(x)
        if self.compress == True:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
        return x
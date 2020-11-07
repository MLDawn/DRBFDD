from torch import nn
import torch.nn.functional as F

# expects the input to be of the shape [batch_size, input_channels, signal_length]



class CNN1D(nn.Module):
    def __init__(self):

        super(CNN1D, self).__init__()
        self.cnn_model = nn.Sequential(
        nn.Conv1d(in_channels=2, out_channels=6, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool1d(5, stride=2),
        nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool1d(5, stride=2))
        self.fc_model = nn.Sequential(
            nn.Linear(656, 120),
            nn.Tanh(),
            nn.Linear(120, 84))

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)

        return x
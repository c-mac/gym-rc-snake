import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    INPUT_CHANNELS = 3

    def __init__(self, board_size):
        super(Network, self).__init__()
        self.first_layer_size = self.INPUT_CHANNELS * (board_size ** 2)
        self.x1 = nn.Linear(self.first_layer_size, 256)
        self.x2 = nn.Linear(256, 4)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.x1(x))
        x = self.x2(x)
        return F.softmin(x, dim=1)

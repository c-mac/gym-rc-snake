import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    INPUT_CHANNELS = 3

    def __init__(self, board_size):
        super(Network, self).__init__()
        # self.conv1 = nn.Conv2d(3, 8, 2, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(8, 16, 2, stride=1, padding=1)
        # self.maxPool = nn.MaxPool2d(3, stride=1, padding=1)
        self.x1 = nn.Linear((board_size ** 2) * self.INPUT_CHANNELS, 512)
        self.x2 = nn.Linear(512, 512)
        self.x3 = nn.Linear(512, 512)
        self.x4 = nn.Linear(512, 4)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x = self.maxPool(self.conv1(x))
        # x = self.maxPool(self.conv2(x))
        x = x.flatten(1)
        x = F.relu(self.x1(x))
        x = F.relu(self.x2(x))
        x = F.relu(self.x3(x))
        x = F.relu(self.x4(x))
        return self.out(x)

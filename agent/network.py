import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    INPUT_CHANNELS = 3

    def __init__(self, board_size):
        super(Network, self).__init__()
        # self.conv1 = nn.Conv2d(3, 8, 2, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(8, 16, 2, stride=1, padding=1)
        # self.maxPool = nn.MaxPool2d(3, stride=1, padding=1)
        self.x1 = nn.Linear(4, 512)
        self.x2 = nn.Linear(512, 2)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x = self.maxPool(self.conv1(x))
        # x = self.maxPool(self.conv2(x))
        x = x.flatten(1)
        x = F.relu(self.x1(x))
        x = F.relu(self.x2(x))
        return self.out(x)


class FC(nn.Module):
    def __init__(self, observation_size, output_size):
        super(FC, self).__init__()
        self.x1 = nn.Linear(observation_size, 256)
        self.x2 = nn.Linear(256, 64)
        self.x3 = nn.Linear(64, output_size)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.x1(x))
        x = F.relu(self.x2(x))
        x = self.x3(x)
        return self.out(x)


class LSTM_FC(nn.Module):
    def __init__(self, observation_size, output_size):
        super(LSTM_FC, self).__init__()
        self.lstm = nn.LSTM(observation_size, 6)
        self.x1 = nn.Linear(6, 512)
        self.x2 = nn.Linear(512, 1024)
        self.x3 = nn.Linear(1024, 512)
        self.x4 = nn.Linear(512, output_size)
        self.out = nn.LogSoftmax(dim=2)

    def forward(self, x):
        # This doesn't quite work?
        x = x.view(len(x), 1, -1)
        x, _ = self.lstm(x)
        x = F.relu(self.x1(x))
        x = F.relu(self.x2(x))
        x = F.relu(self.x4(x))
        return self.out(x)


def fc(observation_size, output_size):
    def fn():
        return FC(observation_size, output_size)

    return fn


def lstm(observation_size, output_size):
    def fn():
        return LSTM_FC(observation_size, output_size)

    return fn

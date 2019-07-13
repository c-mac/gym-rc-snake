import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    INPUT_CHANNELS = 3

    def __init__(self, board_size):
        super(Network, self).__init__()
        self.x1 = nn.Linear(4, 512)
        self.x2 = nn.Linear(512, 2)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.x1(x))
        x = F.relu(self.x2(x))
        return self.out(x)


class FC(nn.Module):
    key = "fc"

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
    key = "lstm"

    def __init__(self, observation_size, output_size):
        super(LSTM_FC, self).__init__()

        # Idk, this seems fine? We can always play with it but if it's working I think
        # it will work with 32 :)
        self.hidden_dim = 32

        self.lstm = nn.LSTM(
            observation_size, self.hidden_dim, batch_first=True, num_layers=1
        )

        # Readout layer
        self.fc = nn.Linear(self.hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.softmax(self.fc(out))
        return (out, hidden)


def fc(observation_size, output_size):
    def fn():
        return FC(observation_size, output_size)

    return fn


def lstm(observation_size, output_size):
    def fn():
        return LSTM_FC(observation_size, output_size)

    return fn

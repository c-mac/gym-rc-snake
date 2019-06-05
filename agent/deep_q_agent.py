import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

LR = 5e-3
GAMMA = 0.9
TAU = 0.2


class NN(nn.Module):
    INPUT_CHANNELS = 3

    def __init__(self, board_size):
        super(NN, self).__init__()
        self.conv = nn.Conv2d(self.INPUT_CHANNELS, 1, 3, padding=1)
        self.x1 = nn.Linear(36, 18)
        self.x2 = nn.Linear(18, 4)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.x1(x))
        x = F.relu(self.x2(x))
        return self.softmax(x)


class DeepQAgent(object):
    """
    Take in board observation and use a Deep Q Network algorithm to decide the next
    observation
    """

    def __init__(self, action_space, board_size):
        self.action_space = action_space
        self.board_size = board_size
        self.qnetwork_local = NN(board_size)
        # local used for current observation --> value of state, action
        self.qnetwork_target = NN(board_size)
        # during learning, target used for next observation --> value of next_state,
        # action in bellman equation then local and target are compared to get a loss,
        # which is used to update local target is updated either with soft update each
        # iteration or is updated to equal the local one every C iterations
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.updates = []

    def act(self, ob, eps=0.0):
        # Take random action (1-eps)% and play according to qnetwork_local
        # best action at this observation otherwise
        if random.random() < eps:
            return self.action_space.sample()
        self.qnetwork_local.eval()
        action_values = self.qnetwork_local(self.obs_to_tensor([ob]))
        self.qnetwork_local.train()
        # print(action_values)
        return np.argmax(action_values.data.numpy())

    def update_value(self, ob, action, reward, next_ob):
        # Keeps track of observations, actions, rewards, next_observations
        # Upon reaching 1000, calls learn() and resets the history to empty
        self.updates.append([ob, action, float(reward), next_ob])
        if len(self.updates) == 1000:
            self.learn()
            self.updates = []

    def learn(self):
        # update NN using all of our updates so far

        # get the value of the next state given the states from the update batch
        # using the target network
        q_targets_next = (
            self.qnetwork_target(self.obs_to_tensor([u[3] for u in self.updates]))
            .max(3)[0]
            .squeeze()
        )

        # rewards from the update batch
        rewards = torch.tensor([u[2] for u in self.updates])

        q_targets = rewards + (GAMMA * q_targets_next)

        # find the actions we took in the updates batch
        actions = torch.tensor([u[1] for u in self.updates])

        # get our q_values given the actions we took in the update batch
        q_expected = self.qnetwork_local(
            self.obs_to_tensor([u[3] for u in self.updates])
        )
        # gather taking the q values for the specific actions that we took
        q_expected = q_expected.squeeze().gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.mse_loss(q_expected, q_targets)
        print(f"LOSS: {loss}")

        # minimize loss on q_expected local network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network parameters to be closer to local network
        # alternate is to set these equal every C steps
        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(
            self.qnetwork_target.parameters(), self.qnetwork_local.parameters()
        ):
            target_param.data.copy_(
                TAU * local_param.data + (1.0 - TAU) * target_param.data
            )

    def obs_to_tensor(self, obs):
        return torch.tensor(
            [
                [
                    [
                        [
                            1.0 if ob[1][-1] == [x, y] else 0.0
                            for x in range(self.board_size)
                            for y in range(self.board_size)
                        ]
                    ],
                    [
                        [
                            1.0 if [x, y] in ob[1][:-1] else 0.0
                            for x in range(self.board_size)
                            for y in range(self.board_size)
                        ]
                    ],
                    [
                        [
                            1.0 if ob[2] == [x, y] else 0.0
                            for x in range(self.board_size)
                            for y in range(self.board_size)
                        ]
                    ],
                ]
                for ob in obs
            ]
        )

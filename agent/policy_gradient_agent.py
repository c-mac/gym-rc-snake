from collections import namedtuple, deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

BATCH_SIZE = 1000
GAMMA = 0.99
LEARNING_RATE = 1e-2


class NN(nn.Module):
    INPUT_CHANNELS = 3

    def __init__(self, board_size):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(self.INPUT_CHANNELS, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)
        self.x1 = nn.Linear(36, 200)
        self.x2 = nn.Linear(200, 4)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.relu(self.x1(x))
        x = self.x2(x)
        return self.softmax(x.squeeze())


class PolicyGradientAgent(object):
    """
    Take in board observation and use a Vanilla Policy Gradient algorithm to decide the
    next observation
    """

    def __init__(self, action_space, board_size, seed):
        random.seed(seed)
        self.action_space = action_space
        self.batch_size = BATCH_SIZE
        self.board_size = board_size
        self.nn = self.load_network(f"savepoints/policy_gradient_{board_size}.pk")
        self.history = []
        self.optimizer = optim.Adam(self.nn.parameters(), lr=LEARNING_RATE)
        self.time_step = 0

    def load_network(self, filename):
        try:
            return torch.load(filename)
        except Exception:
            return NN(self.board_size)

    def update_value(self, ob, action, reward, new_ob, done):
        self.history.append([ob, action, float(reward), new_ob, done])
        if done and len(self.history) > BATCH_SIZE:
            self.learn()

    def learn(self):
        obs = [h[0] for h in self.history]
        actions = [h[1] for h in self.history]
        dones = [h[4] for h in self.history]
        rewards = [h[2] for h in self.history]
        rewards_to_go = torch.zeros(len(self.history))
        rewards_after = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                rewards_after = 0

            reward_here = rewards_after * GAMMA + rewards[i]
            rewards_to_go[i] = reward_here
            rewards_after = reward_here

        rewards_to_go = list(reversed(rewards_to_go))
        # import pdb

        # pdb.set_trace()
        rewards_to_go -= np.mean(rewards_to_go)
        rewards_to_go /= np.std(rewards_to_go) or 0.00001
        rewards_to_go = torch.tensor(rewards_to_go)

        self.optimizer.zero_grad()

        for i in range(len(self.history)):
            probabilities = self.nn(torch.tensor([self.ob_to_tensor(obs[i])]))
            distribution = torch.distributions.Categorical(probabilities)
            surprise = -distribution.log_prob(torch.tensor(actions[i]))
            loss = -probabilities[actions[i]] * rewards_to_go[i]
            if self.time_step % 1000 == 0:
                torch.save(self.nn, f"savepoints/policy_gradient_{self.board_size}.pk")
            if self.time_step % 10000 == 0:
                print(f"loss: {loss}")
            loss.backward()

        self.optimizer.step()

        self.history = []

    def act(self, ob, eps=0.0):
        # Take random action (1-eps)% and play according to qnetwork_local
        # best action at this observation otherwise
        if random.random() < eps:
            return self.action_space.sample()

        # Take an action according to policy

        probs = self.nn(torch.tensor([self.ob_to_tensor(ob)]))
        self.time_step += 1

        try:
            return torch.multinomial(torch.exp(probs), num_samples=1)
        except Exception:
            import pdb

            pdb.set_trace()

    def ob_to_tensor(self, ob):
        return [
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


if __name__ == "__main__":
    agent = PolicyGradientAgent([], 8, 0)

    agent.update_value([], 1, 0, [], 0)
    agent.update_value([], 1, 10, [], 0)
    agent.update_value([], 1, 20, [], 1)
    agent.update_value([], 1, 0, [], 0)
    agent.update_value([], 1, -10, [], 0)
    agent.update_value([], 1, -20, [], 1)

    print(agent.learn())

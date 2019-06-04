import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

GAMMA = 0.9
TAU = 0.2


class NN(nn.Module):
    INPUT_CHANNELS = 3

    def __init__(self, board_size):
        super(NN, self).__init__()
        # self.x = nn.Conv2d(self.INPUT_CHANNELS, 20, 3, padding=1)
        # Need to figure out the input size for this
        self.x1 = nn.Linear(board_size ** 2 * self.INPUT_CHANNELS, 200)
        self.x2 = nn.Linear(200, 4)

    def forward(self, x):
        x = F.relu(self.x1(x))
        x = F.relu(self.x2(x))
        return F.softmax(x)


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
        # during learning, target used for next observation --> value of next_state, action in bellman equation
        # then local and target are compared to get a loss, which is used to update local
        # target is updated either with soft update each iteration or 
        # is updated to equal the local one every C iterations
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.03)
        self.updates = []

    def act(self, ob, eps=0.0):
    #Take random action (1-eps)% and play according to qnetwork_local 
    #best action at this observation otherwise
        if random.random() < eps:
            self.qnetwork_local.eval()
            action_values = self.qnetwork_local(
                torch.tensor(self.observation_to_tensor(ob))
            )
            self.qnetwork_local.train()
            return np.argmax(action_values.data.numpy())

        return self.action_space.sample()

    def update_value(self, ob, action, reward, next_ob):
    #Keeps track of observations, actions, rewards, next_observations
    #Upon reaching 1000, calls learn() and resets the history to empty
        self.updates.append([ob, action, float(reward), next_ob])
        if len(self.updates) == 1000:
            self.learn()
            self.updates = []

    def learn(self):
        # update NN using all of our updates so far


        #get the value of the next state given the states from the update batch 
        #using the target network
        q_targets_next = (
            self.qnetwork_target(
                torch.tensor(
                    list(
                        map(
                            lambda update: self.observation_to_tensor(update[3]),
                            self.updates,
                        )
                    )
                )
            )
            .max(1)[
                0
            ]  # gets the maximum value for next_ob according to qnetwork_target
            .unsqueeze(1)  # adds extra dimension
        )

        #rewards from the update batch
        rewards = torch.tensor([u[2] for u in self.updates])

        q_targets = rewards + (GAMMA * q_targets_next.t())

        #find the actions we took in the updates batch
        actions = torch.tensor([u[1] for u in self.updates])

        #get our q_values given the actions we took in the update batch
        q_expected = self.qnetwork_local(
            torch.tensor(
                list(
                    map(
                        lambda update: self.observation_to_tensor(update[0]),
                        self.updates,
                    )
                )
            )
        )
        # gather taking the q values for the specific actions that we took
        q_expected = q_expected.gather(1, actions.unsqueeze(1))

        loss = F.mse_loss(q_expected, q_targets)

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

    def best_value_and_action(self, ob):
        import pdb

        pdb.set_trace()
        return 0, 0

    def observation_to_tensor(self, ob):
        _last_move, snake, food = ob
        head = snake[-1]
        tail = snake[:-1]
        return np.array(
            [
                [
                    1.0 if [x, y] in tail else 0.0
                    for x in range(self.board_size)
                    for y in range(self.board_size)
                ],
                [
                    1.0 if [x, y] == head else 0.0
                    for x in range(self.board_size)
                    for y in range(self.board_size)
                ],
                [
                    1.0 if [x, y] == food else 0.0
                    for x in range(self.board_size)
                    for y in range(self.board_size)
                ],
            ],
            dtype=np.float32,
        ).flatten()

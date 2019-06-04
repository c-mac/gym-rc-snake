import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

GAMMA = 0.9

class NN(nn.Module):
    INPUT_CHANNELS = 3

    def __init__(self, board_size):
        super(NN, self).__init__()
        # self.x = nn.Conv2d(self.INPUT_CHANNELS, 20, 3, padding=1)
        # Need to figure out the input size for this
        self.x1 = nn.Linear(board_size ** 2 * self.INPUT_CHANNELS, 200)
        self.x2 = nn.Linear(200, 4)

    def forward(self, x):
        x = self.x1(x)
        x = self.x2(x)
        return F.softmax(x)


class DeepQAgent(object):
    """
    Take in board obersvation and use a Deep Q Network algorithm to decide the next
    observation
    """

    def __init__(self, action_space, board_size):
        self.action_space = action_space
        self.board_size = board_size
        self.qnetwork_local = NN(board_size)
        self.qnetwork_target = NN(board_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.03)
        self.updates = []

    def act(self, ob, eps = 0.):
        if random.random() < eps:
            self.qnetwork_local.eval()
            action_values = self.qnetwork_local(torch.tensor(self.observation_to_tensor(ob)))
            self.qnetwork_local.train()
            return np.argmax(action_values.data.numpy())

        return self.action_space.sample()

    def update_value(self, ob, action, reward, next_ob):
        self.updates.append([ob, action, float(reward), next_ob])
        if len(self.updates) == 10:
            self.learn()
            self.updates = []

    def learn(self):
        # update NN using all of our updates so far
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
            .max(1)[0] #gets the maximum value for next_ob according to qnetwork_target
            .unsqueeze(1) #adds extra dimension
        )

        rewards = torch.tensor([u[2] for u in self.updates])

        try:
            q_targets = rewards + (GAMMA * q_targets_next.t())
        except Exception:
            import pdb; pdb.set_trace()

        actions = torch.tensor([u[1] for u in self.updates])
        q_expected = self.qnetwork_local(torch.tensor(
            list(
                map(
                    lambda update: self.observation_to_tensor(update[0]),
                    self.updates,
                )
            )
        ))
        #gather taking the state values for the actions that we took
        q_expected = q_expected.gather(1,actions.unsqueeze(1)) 

        loss = F.mse_loss(q_expected, q_targets)

        #minimize loss on q_expected local network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #could do "soft" update here for target network


    def best_value_and_action(self, ob):
        import pdb; pdb.set_trace()
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

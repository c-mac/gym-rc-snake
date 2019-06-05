from collections import namedtuple, deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
GAMMA = 0.9
LEARNING_RATE = 1e-3
TAU = 1e-3


class NN(nn.Module):
    INPUT_CHANNELS = 3

    def __init__(self, board_size):
        super(NN, self).__init__()
        self.conv = nn.Conv2d(self.INPUT_CHANNELS, 1, 3, padding=1)
        self.x1 = nn.Linear(36, 18)
        self.x2 = nn.Linear(18, 4)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.x1(x))
        return self.x2(x)


class DeepQAgent(object):
    """
    Take in board observation and use a Deep Q Network algorithm to decide the next
    observation
    """

    def __init__(self, action_space, board_size, seed):
        random.seed(seed)
        self.action_space = action_space
        self.batch_size = BATCH_SIZE
        self.board_size = board_size
        self.qnetwork_local = self.load_network("savepoints/local.pk")
        # local used for current observation --> value of state, action
        self.qnetwork_target = self.load_network("savepoints/target.pk")
        # during learning, target used for next observation --> value of next_state,
        # action in bellman equation then local and target are compared to get a loss,
        # which is used to update local target is updated either with soft update each
        # iteration or is updated to equal the local one every C iterations
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)
        # replay memory
        self.memory = ReplayBuffer(action_space, BUFFER_SIZE, BATCH_SIZE, seed)
        self.time_step = 0

    def load_network(self, filename):
        try:
            return torch.load(filename)
        except Exception:
            return NN(self.board_size)

    def act(self, ob, eps=0.0):
        # Take random action (1-eps)% and play according to qnetwork_local
        # best action at this observation otherwise
        if random.random() < eps:
            return self.action_space.sample()
        self.qnetwork_local.eval()

        action_values = self.qnetwork_local(torch.tensor([self.ob_to_tensor(ob)]))
        self.qnetwork_local.train()
        return np.argmax(action_values.data.numpy())

    def update_value(self, ob, action, reward, next_ob, done):
        # Keeps track of observations, actions, rewards, next_observations
        # Upon reaching 1000, calls learn() and resets the history to empty

        self.memory.add(
            self.ob_to_tensor(ob),
            action,
            float(reward),
            self.ob_to_tensor(next_ob),
            done,
        )
        self.time_step += 1
        if len(self.memory) > self.batch_size and self.time_step % 4 == 0:
            memories = self.memory.sample()
            self.learn(memories)

    def learn(self, memories):
        # get the value of the next state given the states from the memories batch
        # using the target network
        q_targets_next = self.qnetwork_target(memories[3]).max(3)[0].squeeze()

        # rewards from the memories batch
        rewards = memories[2].squeeze()

        q_targets = rewards + GAMMA * q_targets_next * (1 - memories[4].squeeze())

        # find the actions we took in the memories batch
        actions = memories[1]

        # get our q_values given the actions we took in the memories batch
        q_expected = self.qnetwork_local(memories[0])

        # gather taking the q values for the specific actions that we took
        q_expected = q_expected.squeeze().gather(1, actions).squeeze()

        loss = F.mse_loss(q_expected, q_targets)
        if self.time_step % 1000 == 0:
            print(f"LOSS: {loss}")

        # minimize loss on q_expected local network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.time_step % 1000 == 0:
            torch.save(self.qnetwork_local, "savepoints/local.pk")
            torch.save(self.qnetwork_target, "savepoints/target.pk")

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


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.tensor([e.state for e in experiences if e is not None])
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long()
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float()
        next_states = torch.tensor([e.next_state for e in experiences if e is not None])
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""

        return len(self.memory)

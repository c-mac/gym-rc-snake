from gym_rc_snake.envs.snake_env import SnakeRCEnv
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.optim as optim
from torch.distributions import Categorical

from agent.network import Network

Experience = namedtuple(
    "Experience",
    field_names=["observation", "action", "reward", "next_observation", "done"],
)

GAMMA = 0.8

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
MOVE_OPPOSITE = {LEFT: RIGHT, DOWN: UP, RIGHT: LEFT, UP: DOWN}


class PolicyGradientAgent:
    """
    Use Vanilla Policy Gradient
    """

    def __init__(self, action_space, board_size, seed=12345, network_name=None):
        random.seed(seed)
        if network_name:
            self.network_name = (
                f"savepoints/{network_name}-policy_gradient_"
                f"{board_size}x{board_size}.pk"
            )
        else:
            self.network_name = None
        self.action_space = action_space
        self.batch_size = 1024
        self.board_size = board_size
        self.network = self.load_network(self.network_name)

        self.history = []
        self.graph_env = SnakeRCEnv()
        self.optimizer = optim.SGD(self.network.parameters(), lr=1e-2)
        self.t = 0

    def load_network(self, filename):
        if not filename:
            return Network(self.board_size)

        try:
            network = torch.load(filename)
            print(f"Successfully loaded network from file: {filename}")
            return network
        except Exception:
            print(f"Could not load network from {filename}, creating a new one")
            return Network(self.board_size)

    def act(self, observation):
        """
        Take in an observation
        Run it through the network to get an action distribution
        Sample from that action distribution and return the result
        """
        self.t += 1

        probabilities = self.probabilities(observation)

        probabilities *= self.action_mask(observation[0])

        return torch.multinomial(probabilities, num_samples=1)[0]

    def action_mask(self, last_move):
        action_mask = torch.ones(self.action_space.n)
        action_mask[MOVE_OPPOSITE[last_move]] = 0.0
        return action_mask

    def update_value(self, *args, **kwargs):
        return self.remember(*args, **kwargs)

    def remember(self, observation, action, reward, next_observation, done):
        self.history.append(
            Experience(observation, action, reward, next_observation, done)
        )

        if len(self.history) > self.batch_size and done:
            self.learn_from_history(self.history)
            self.history = []

    def learn_from_history(self, history, log=False):
        rewards_to_go = torch.zeros(len(history))
        rewards = [h.reward for h in history]
        rewards_after = 0
        for i in reversed(range(len(rewards))):
            if history[i].done:
                rewards_after = 0

            reward_here = rewards_after * GAMMA + rewards[i]
            rewards_to_go[i] = reward_here
            rewards_after = reward_here

        if log:
            print(rewards_to_go)

        rewards_to_go -= rewards_to_go.mean()
        std = rewards_to_go.std()

        if std < 0.001:
            print("Not training on a set of rewards that are all the same")
            return

        rewards_to_go /= std
        if log:
            print(rewards_to_go)

        to_train_from = [
            (history[i].observation, history[i].action, rewards_to_go[i])
            for i in range(len(history))
        ]

        self.optimizer.zero_grad()

        probs = self.network(
            torch.stack(
                [self.observation_as_network_input(t[0]) for t in to_train_from]
            )
        )
        if log:
            print(probs[0])
        log_probs = torch.zeros(len(probs))
        for i in range(len(probs)):
            log_probs[i] = Categorical(probs[i]).log_prob(to_train_from[i][1])

        rewards_to_go = torch.tensor([t[2] for t in to_train_from])
        loss = (-log_probs * rewards_to_go).mean()

        if log:
            print(loss)

        loss.backward()

        self.optimizer.step()

        if log:
            print(f"Saving network to file {self.network_name}")
        torch.save(self.network, self.network_name)

    def probabilities(self, observation):
        return self.network(self.observation_as_network_input(observation)[None])

    def observation_as_network_input(self, ob):
        return torch.tensor(
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
        )

    def plot_motion_graph(self, now):

        X = np.arange(0, self.board_size, 1)
        Y = np.arange(0, self.board_size, 1)

        U = [[0.0 for _ in Y] for _ in X]
        V = [[0.0 for _ in X] for _ in Y]
        for x in X:
            for y in Y:
                self.graph_env.snake = [[x, y]]
                self.graph_env.food = [2, 2]
                ob = self.graph_env.observation()
                probs = self.network(self.observation_as_network_input(ob)[None])[0]
                U[x][y] = probs[LEFT].item() - probs[RIGHT].item()
                V[x][y] = probs[UP].item() - probs[DOWN].item()

        fig, ax = plt.subplots()
        ax.quiver(X, Y, U, V)

        plt.savefig(f"monitor/{now}-map.png")
        plt.close("all")

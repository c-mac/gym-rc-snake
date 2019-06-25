from gym_rc_snake.envs.snake_env import SnakeRCEnv
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.optim as optim

from agent.network import Network

Experience = namedtuple(
    "Experience",
    field_names=["observation", "action", "reward", "next_observation", "done"],
)

GAMMA = 0.99

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
MOVE_OPPOSITE = {LEFT: RIGHT, DOWN: UP, RIGHT: LEFT, UP: DOWN}


class PPOAgent:
    """
    Use PPO
    """

    def __init__(self, action_space, board_size, seed=None, network_name=None):
        if seed:
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
        self.old_network = Network(self.board_size)
        self.old_network.load_state_dict(self.network.state_dict())

        self.history = []
        self.graph_env = SnakeRCEnv()
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        self.t = 0

    def load_network(self, filename):
        if not filename:
            return Network(self.board_size)

        try:
            network = torch.load(filename)
            print(f"Successfully loaded network from file: {filename}")
            return network
        except Exception as e:
            print(f"Could not load network from {filename}, creating a new one")
            return Network(self.board_size)

    def act(self, observation):
        """
        Take in an observation
        Run it through the network to get an action distribution
        Sample from that action distribution and return the result
        """

        with torch.no_grad():
            self.t += 1

            probabilities = self.probabilities(observation)
            if self.t % 1000 == 0:
                print(probabilities)

            # probabilities *= self.action_mask(observation[0])

            if probabilities.sum() == 0.0:
                probabilities += 1.0

            return torch.multinomial(probabilities, num_samples=1)[0].item()

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

    def ppo_history(self, state, action_taken, reward, clip_value=0.1):
        input = torch.stack(
            list(map(self.cartpole_observation_as_network_input, state))
        )
        action_taken = torch.tensor(action_taken).view(-1, 1)
        reward = torch.stack(reward)

        # How likely our action was under our old policy
        old_probs = torch.exp(self.old_network(input)).gather(1, action_taken)

        # How likely it was that we would take this action with our current policy
        new_probs = torch.exp(self.network(input)).gather(1, action_taken)

        # this is r_t_theta in the PPO paper
        ratio = (new_probs / old_probs).squeeze()

        # this is the left side of the min params
        # Note that we are using reward here instead of advantage, it's simpler but it
        # should work about as well
        surr1 = ratio * reward
        # This is our clipped, right hand side!
        surr2 = torch.clamp(ratio, min=1 - clip_value, max=1 + clip_value) * reward

        combined = torch.cat((surr1, surr2)).view(2, len(surr1))
        return -(combined.min(0).values).mean()

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

        for _ in range(1):
            loss = self.ppo_history(
                [(history[i].observation) for i in range(len(history))],
                [history[i].action for i in range(len(history))],
                [rewards_to_go[i] for i in range(len(history))],
            )

            self.optimizer.zero_grad()

            if torch.isnan(loss):
                import pdb

                pdb.set_trace()

            print(f"LOSS: {loss}")

            self.old_network.load_state_dict(self.network.state_dict())

            loss.backward()

            self.optimizer.step()

        if log:
            print(f"Saving network to file {self.network_name}")
        torch.save(self.network, self.network_name)

    def probabilities(self, observation):
        return torch.exp(
            self.network(self.cartpole_observation_as_network_input(observation)[None])
        )[0]

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

    def cartpole_observation_as_network_input(self, ob):
        return torch.tensor(ob).float()

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
                probs = self.probabilities(ob)
                U[x][y] = probs[LEFT].item() - probs[RIGHT].item()
                V[x][y] = probs[UP].item() - probs[DOWN].item()

        fig, ax = plt.subplots()
        ax.quiver(X, Y, U, V)

        plt.savefig(f"monitor/{now}-map.png")
        plt.close("all")

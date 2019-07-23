from collections import namedtuple

import torch
import torch.optim as optim

Experience = namedtuple(
    "Experience",
    field_names=["observation", "action", "reward", "next_observation", "done"],
)


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
MOVE_OPPOSITE = {LEFT: RIGHT, DOWN: UP, RIGHT: LEFT, UP: DOWN}


GAMMA = 0.99


def binary_to_number(b):
    place = 0
    total = 0
    for bit in b:
        if bit:
            total += 2 ** place
        place += 1

    return total


class CellularAutomataAgent:
    """
    Use cellular automata
    """

    def __init__(self, action_space, network_fn, network_name=None):
        self.observation_space = 64  # (actually, it's 2 ** the observation space)
        self.action_space = action_space
        self.t = 0
        # This is almost certainly a bad rule set
        self.RULES = [1] * 64
        self.memory = []

    def act(self, observation):
        """
        Take in an observation
        Look up its value in our rule set
        Return that value
        """
        self.t += 1
        return self.RULES[binary_to_number(observation)]

    def learn_from_memory(self, memory, log=False):
        rewards_to_go = torch.zeros(len(memory))
        rewards = [h.reward for h in memory]
        rewards_after = 0
        for i in reversed(range(len(rewards))):
            if memory[i].done:
                rewards_after = 0

            reward_here = rewards_after * GAMMA + rewards[i]
            rewards_to_go[i] = reward_here
            rewards_after = reward_here

        outcomes = torch.zeros(len(self.RULES))

        for (i, m) in enumerate(memory):
            outcomes[binary_to_number(m.observation)] += rewards_to_go[i]

        for (i, m) in enumerate(outcomes):
            if outcomes[i] < 0:
                self.RULES[i] = (self.RULES[i] + 1) % 3

    def update_value(self, *args, **kwargs):
        return self.remember(*args, **kwargs)

    def remember(self, observation, action, reward, next_observation, done):
        self.memory.append(
            Experience(observation, action, reward, next_observation, done)
        )

        if done:
            self.learn_from_memory(self.memory)
            self.memory = []

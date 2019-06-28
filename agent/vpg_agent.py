from collections import namedtuple

import torch
import torch.optim as optim

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


class VPGAgent:
    """
    Use VPG
    """

    def __init__(self, action_space, network_fn, network_name=None):
        self.action_space = action_space
        self.batch_size = 256
        self.network_fn = network_fn
        self.network_name = network_name

        self.network = self.load_network(self.network_name)

        self.history = []
        self.optimizer = optim.Adam(self.network.parameters(), lr=5e-3)
        self.t = 0

    def load_network(self, filename):
        if not filename:
            return self.network_fn()

        try:
            network = torch.load(filename)
            print(f"Successfully loaded network from file: {filename}")
            return network
        except Exception:
            print(f"Could not load network from {filename}, creating a new one")
            return self.network_fn()

    def act(self, observation):
        """
        Take in an observation
        Run it through the network to get an action distribution
        Sample from that action distribution and return the result
        """

        with torch.no_grad():
            self.t += 1

            probabilities = self.probabilities(observation)

            if self.t % 5000 == 0:
                print(probabilities)

            if probabilities.sum() == 0.0:
                probabilities += 1.0

            return torch.multinomial(probabilities, num_samples=1)[0].item()

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

        # if log:
        # print(rewards_to_go)

        states = [history[i].observation for i in range(len(history))]
        actions = torch.tensor([history[i].action for i in range(len(history))]).view(
            -1, 1
        )
        rewards = torch.tensor([history[i].reward for i in range(len(history))])

        logits = self.network(
            torch.stack(list(map(lambda x: torch.tensor(x).float(), states)))
        )

        loss = (logits.gather(1, actions) * rewards_to_go).mean()

        self.optimizer.zero_grad()

        # if log:
        print(f"LOSS: {loss}")

        loss.backward()

        self.optimizer.step()
        # print([x.std().item() for x in self.network.parameters()])

        if log:
            print(f"Saving network to file {self.network_name}")
        torch.save(self.network, self.network_name)

    def probabilities(self, observation):
        logits = self.network(torch.tensor(observation).float()[None])
        return torch.exp(logits)[0]

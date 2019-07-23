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


class LstmPpoAgent:
    """
    Use PPO with an LSTM
    """

    def __init__(self, action_space, network_fn, network_name=None):
        self.action_space = action_space
        self.batch_size = 1024
        self.network_fn = network_fn
        self.network_name = network_name

        self.network = self.load_network(self.network_name)
        self.old_network = network_fn()
        self.old_network.load_state_dict(self.network.state_dict())

        self.history = []
        # 32 just happens to be the size of our hidden stuff for LSTM purposes
        # The number is kind of arbitrary, it seems big enough to hold important
        # information, small enough not to slow down the training
        self.hidden = (torch.zeros(1, 1, 32), torch.zeros(1, 1, 32))
        self.episode_history = []
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
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
            self.episode_history.append(observation)
            self.t += 1

            probabilities = self.probabilities(observation)

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

        if done:
            self.episode_history = []
            self.hidden = (torch.zeros(1, 1, 32), torch.zeros(1, 1, 32))

        if len(self.history) > self.batch_size and done:
            self.learn_from_history(self.history)
            self.history = []

    def ppo(self, state, action_taken, reward, clip_value=0.1):
        input = torch.stack(list(map(lambda x: torch.tensor(x).float(), state)))
        action_taken = torch.tensor(action_taken).view(-1, 1)
        reward = torch.stack(reward)

        # How likely our action was under our old policy

        old_probs = torch.exp(
            self.old_network(input[None], self.hidden)[0].view(len(input), -1)
        ).gather(1, action_taken)

        # How likely it was that we would take this action with our current policy
        new_probs = torch.exp(
            self.network(input[None], self.hidden)[0].view(len(input), -1)
        ).gather(1, action_taken)

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
        # So, now we want to learn, and we do that *by episode*
        # This is because for each episode we will get the correct context, but our
        # context has to reset at the end of each episode.

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

        # PPO theoretically allows you to train multiple times on the same data. Let's
        # not worry about that right now because it's not necessary when we have
        # environments that are run quite cheaply like snake or cartpole.
        for _ in range(2):
            # OK, here's a little switcheroo to simplify some stuff for now. Let's only
            # train on the first episode
            history_by_episode = []
            episode = []

            for i in range(len(history)):
                if history[i].done:
                    history_by_episode.append(episode)
                    episode = []
                else:
                    h = history[i]
                    episode.append(
                        Experience(
                            observation=h.observation,
                            action=h.action,
                            reward=rewards_to_go[i],
                            next_observation=None,
                            done=None,
                        )
                    )

            for history in history_by_episode:
                loss = self.ppo(
                    [(history[i].observation) for i in range(len(history))],
                    [history[i].action for i in range(len(history))],
                    [history[i].reward for i in range(len(history))],
                )

                self.optimizer.zero_grad()

                if log:
                    print(f"LOSS: {loss}")

                loss.backward()

        self.old_network.load_state_dict(self.network.state_dict())
        self.optimizer.step()

        if log:
            print(f"Saving network to file {self.network_name}")
        torch.save(self.network, self.network_name)

    def probabilities(self, observation):
        (logits, hidden) = self.network(
            torch.tensor(observation).float().view(1, 1, -1), self.hidden
        )
        self.hidden = hidden

        if self.t % 1000 == 0:
            print(torch.exp(logits)[0])

        return torch.exp(logits)[0]

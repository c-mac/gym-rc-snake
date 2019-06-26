import json
import numpy as np


class Stats:
    def __init__(self, name):
        self.file = open(f"monitor/{name}_stats.json", "w")
        self.top_level_stats = []

    def record(self, rewards, lengths):
        """ From
        https://spinningup.openai.com/en/latest/spinningup/spinningup.html#learn-by-doing:

        I personally like to look at the mean/std/min/max for cumulative rewards,
        episode lengths, and value function estimates, along with the losses for the
        objectives, and the details of any exploration parameters (like mean entropy for
        stochastic policy optimization, or current epsilon for epsilon-greedy as in
        DQN)."""
        # So we're going to get in a whole bunch of stats and we want to pull out the
        # mean, std, min and max for all of them!
        reward = self.meta_stats(rewards)
        length = self.meta_stats(lengths)
        stats = {"reward": reward, "length": length}
        print(
            f"""
\tREWARD\t\tLENGTH
MEAN\t{round(reward['mean'], 2)}\t\t{round(length['mean'], 2)}
STD\t{round(reward['std'], 2)}\t\t{round(length['std'], 2)}
MIN\t{round(reward['min'], 2)}\t\t{round(length['min'], 2)}
MAX\t{round(reward['max'], 2)}\t\t{round(length['max'], 2)}
            """
        )
        self.top_level_stats.append(stats)
        self.file.seek(0)
        self.file.write(json.dumps(self.top_level_stats))
        self.file.flush()

    def meta_stats(self, stats):
        if not stats:
            return {"mean": float(0), "std": float(0), "min": float(0), "max": float(0)}
        return {
            "mean": float(np.mean(stats)),
            "std": float(np.std(stats)),
            "min": float(np.min(stats)),
            "max": float(np.max(stats)),
        }

import json
import numpy as np
import matplotlib.pyplot as plt


stats = json.loads(open("monitor/1560716697_stats.json").read())

x = np.arange(len(stats))
reward = [s["reward"]["mean"] for s in stats]
reward_std = [s["reward"]["std"] / 10.0 for s in stats]
food = [s["food"]["mean"] for s in stats]
food_std = [s["food"]["std"] / 10.0 for s in stats]
length = [s["length"]["mean"] for s in stats]
length_std = [s["length"]["std"] / 10.0 for s in stats]

fig, ax = plt.subplots()
ax.errorbar(x, reward, yerr=reward_std)
ax.errorbar(x, food, yerr=food_std)
ax.errorbar(x, length, yerr=length_std)
plt.show()

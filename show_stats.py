import json
import sys
import numpy as np
import matplotlib.pyplot as plt


WINDOW_SIZE = 5
stats = json.loads(open(sys.argv[1]).read())
x = np.arange(len(stats))
reward = [s["reward"]["mean"] for s in stats]
reward_std = [s["reward"]["std"] / 10.0 for s in stats]
# food = [s["food"]["mean"] for s in stats]
# food_std = [s["food"]["std"] / 10.0 for s in stats]
length = [s["length"]["mean"] for s in stats]
length_std = [s["length"]["std"] / 10.0 for s in stats]


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


f, ax1 = plt.subplots(1, 1)
if len(x) < WINDOW_SIZE ** 2:
    ax1.title.set_text("Reward")
    ax1.plot(x, reward, "o-", markevery=10)
    # ax2.title.set_text("Food")
    # ax2.plot(x, food, "o-", markevery=10)
    # ax3.title.set_text("Steps")
    # ax3.plot(x, length, "o-", markevery=10)
else:
    ax1.title.set_text("Reward")
    ax1.plot(
        x[WINDOW_SIZE:-WINDOW_SIZE],
        smooth(reward, WINDOW_SIZE)[WINDOW_SIZE:-WINDOW_SIZE],
        "o-",
        markevery=10,
    )
    # ax2.title.set_text("Food")
    # ax2.plot(
    #     x[WINDOW_SIZE:-WINDOW_SIZE],
    #     smooth(food, WINDOW_SIZE)[WINDOW_SIZE:-WINDOW_SIZE],
    #     "o-",
    #     markevery=10,
    # )
    # ax3.title.set_text("Steps")
    # ax3.plot(
    #     x[WINDOW_SIZE:-WINDOW_SIZE],
    #     smooth(length, WINDOW_SIZE)[WINDOW_SIZE:-WINDOW_SIZE],
    #     "o-",
    #     markevery=10,
    # )

plt.tight_layout()
plt.show()

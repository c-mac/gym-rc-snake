import gym
import json
import time
import torch
import matplotlib.pyplot as plt
import numpy as np

from gym import logger
from agent.policy_gradient_agent import PolicyGradientAgent

from gym.envs.registration import register

register(id="snake-rc-v0", entry_point="gym_rc_snake.envs:SnakeRCEnv")

top_level_stats = []


def record_stats(rewards, lengths, foods, stats_file=None):
    """
    From
    https://spinningup.openai.com/en/latest/spinningup/spinningup.html#learn-by-doing:

    I personally like to look at the mean/std/min/max for cumulative rewards,
    episode lengths, and value function estimates, along with the losses for the
    objectives, and the details of any exploration parameters (like mean entropy for
    stochastic policy optimization, or current epsilon for epsilon-greedy as in DQN).
    """
    # So we're going to get in a whole bunch of stats and we want to pull out the mean,
    # std, min and max for all of them!
    stats = {
        "reward": meta_stats(rewards),
        "length": meta_stats(lengths),
        "food": meta_stats(foods),
    }
    print(stats)
    top_level_stats.append(stats)
    if stats_file:
        stats_file.seek(0)
        stats_file.write(json.dumps(top_level_stats))
        stats_file.flush()


def meta_stats(stats):
    return {
        "mean": float(np.mean(stats)),
        "std": float(np.std(stats)),
        "min": float(np.min(stats)),
        "max": float(np.max(stats)),
    }


if __name__ == "__main__":
    GRAPH = True
    RENDER = False
    TRAIN = True

    logger.set_level(logger.DEBUG)

    env = gym.make("snake-rc-v0", render=RENDER)
    agent = PolicyGradientAgent(env.action_space, env.board_size, 12345, "main")

    done = False
    reward = 0
    rewards_iter = 0
    episode_rewards = []
    sum_steps = []
    step_rewards = []
    test_episodes = 500
    training_episodes = 500
    total_episodes = 0
    now = int(time.time())
    stats_file = open(f"monitor/{now}_stats.json", "w")

    while True:
        with torch.no_grad():
            episode_stats = []
            for t in range(test_episodes):
                ob = env.reset()
                episode_length = 0
                episode_food = 0
                episode_reward = 0
                while True:
                    action = agent.act(ob)
                    ob, reward, done, info = env.step(action)
                    if episode_length > 200:
                        reward = -10
                        done = True
                    if reward == 1.0:
                        episode_food += 1
                    episode_reward += reward
                    episode_length += 1
                    if RENDER:
                        env.render()
                        time.sleep(0.05)
                    if done:
                        episode_stats.append(
                            (episode_reward, episode_length, episode_food)
                        )
                        env.close()
                        break

            record_stats(
                [e[0] for e in episode_stats],
                [e[1] for e in episode_stats],
                [e[2] for e in episode_stats],
                stats_file,
            )

        if GRAPH:
            rewards_iter += 1
            episode_rewards.append(episode_reward / test_episodes)
            step_rewards.append(episode_reward / episode_length)
            sum_steps.append(episode_length / test_episodes)
            if rewards_iter % 1 == 0:
                plt.subplot(2, 1, 1)
                plt.plot(np.arange(len(episode_rewards)), episode_rewards)
                plt.ylabel("Avg. Reward per Episode")

                plt.subplot(2, 1, 2)
                plt.plot(np.arange(len(sum_steps)), sum_steps)
                plt.ylabel("Avg. Steps per Episode")

                plt.xlabel("Test Episode Batches")

                plt.savefig(f"monitor/{now}-training-progress.png")

                agent.plot_motion_graph(now)
                plt.close("all")

        if TRAIN:
            for t in range(training_episodes):
                episode_length = 0
                ob = env.reset()
                while True:
                    action = agent.act(ob)
                    new_ob, reward, done, info = env.step(action)
                    if episode_length > 1000:
                        reward = -10
                        done = True
                    agent.update_value(ob, action, reward, new_ob, done)
                    ob = new_ob
                    episode_length += 1
                    if done:
                        env.close()
                        break

    env.close()

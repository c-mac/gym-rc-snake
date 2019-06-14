import torch
import time
import gym
from gym import logger
from agent.policy_gradient_agent import PolicyGradientAgent
import matplotlib.pyplot as plt
import numpy as np

from gym.envs.registration import register

register(id="snake-rc-v0", entry_point="gym_rc_snake.envs:SnakeRCEnv")


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

    while True:
        with torch.no_grad():
            episode_reward = 0
            episode_total_steps = 0
            for t in range(test_episodes):
                ob = env.reset()
                episode_length = 0
                while True:
                    action = agent.act(ob)
                    ob, reward, done, info = env.step(action)
                    if episode_length > 200:
                        reward = -10
                        done = True
                    episode_reward += reward
                    episode_total_steps += 1
                    episode_length += 1
                    if RENDER:
                        env.render()
                        time.sleep(0.05)
                    if done:
                        env.close()
                        break

            print(f"EPISODE REWARD: {episode_reward}")
            print(f"EPISODE TOTAL STEPS: {episode_total_steps / float(test_episodes)}")

            if GRAPH:
                rewards_iter += 1
                episode_rewards.append(episode_reward / test_episodes)
                step_rewards.append(episode_reward / episode_total_steps)
                sum_steps.append(episode_total_steps / test_episodes)
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

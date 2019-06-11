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
    RENDER = True
    TRAIN = False

    logger.set_level(logger.DEBUG)

    env = gym.make("snake-rc-v0", render=RENDER)
    agent = PolicyGradientAgent(env.action_space, env.board_size, 12345, "main")

    done = False
    reward = 0
    rewards_iter = 0
    sum_rewards = []
    sum_steps = []
    test_episodes = 200
    training_episodes = 1000
    total_episodes = 0
    now = int(time.time())

    while True:
        with torch.no_grad():
            episode_reward = 0
            episode_total_steps = 0
            for t in range(test_episodes):
                ob = env.reset()
                for i in range(200):
                    action = agent.act(ob)
                    ob, reward, done, info = env.step(action)
                    episode_reward += reward
                    episode_total_steps += 1
                    if RENDER:
                        env.render()
                        time.sleep(0.02)
                    if done:
                        env.close()
                        break

            print(f"EPISODE REWARD: {episode_reward}")
            print(f"EPISODE TOTAL STEPS: {episode_total_steps / float(test_episodes)}")

            if GRAPH:
                rewards_iter += 1
                sum_rewards.append(episode_reward / episode_total_steps)
                sum_steps.append(episode_total_steps / test_episodes)
                if rewards_iter % 1 == 0:
                    plt.subplot(2, 1, 1)
                    plt.plot(np.arange(len(sum_rewards)), sum_rewards)
                    plt.ylabel("Avg. Reward per Step")

                    plt.subplot(2, 1, 2)
                    plt.plot(np.arange(len(sum_steps)), sum_steps)
                    plt.ylabel("Avg. Steps per Episode")

                    plt.xlabel("Test Episode Batches")

                    plt.savefig(f"monitor/{now}-training-progress.png")

                    agent.plot_motion_graph(now)
                    plt.close("all")

        if TRAIN:
            for t in range(training_episodes):
                ob = env.reset()
                for i in range(100):
                    action = agent.act(ob)
                    new_ob, reward, done, info = env.step(action)
                    agent.update_value(ob, action, reward, new_ob, done)
                    ob = new_ob
                    if done:
                        env.close()
                        break

    env.close()

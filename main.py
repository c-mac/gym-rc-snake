import gym
import time
import torch

from agent.policy_gradient_agent import PolicyGradientAgent
from agent.ppo_agent import PPOAgent
from agent.network import fc, lstm
from stats import Stats

from gym import logger
from wrappers.snake import BoardOnly, SnakePerspective
from gym.envs.registration import register

register(id="snake-rc-v0", entry_point="gym_rc_snake.envs:SnakeRCEnv")


def test(env, agent, num_episodes, stats, render=False):
    with torch.no_grad():
        episode_stats = []
        for t in range(num_episodes):
            episode_stats.append(one_episode(env, agent, render=render))
    stats.record([e[0] for e in episode_stats], [e[1] for e in episode_stats])


def one_episode(env, agent, pause=0.02, train=False, render=False):
    cum_reward = 0
    length = 0
    ob = env.reset()
    old_ob = ob
    while True:
        action = agent.act(ob)
        ob, reward, done, info = env.step(action)

        if train:
            agent.update_value(old_ob, action, reward, ob, done)

        old_ob = ob

        cum_reward += reward
        length += 1

        if render:
            env.render()
            time.sleep(pause)

        if done:
            env.close()
            return (cum_reward, length)


def train(env, agent, num_episodes):
    for t in range(num_episodes):
        one_episode(env, agent, train=True)


if __name__ == "__main__":
    logger.set_level(logger.DEBUG)
    RENDER = True
    BOARD_SIZE = 8

    env = gym.make("snake-rc-v0", render=RENDER)
    env.reset()
    env = SnakePerspective(env)
    network_fn = fc(6, env.action_space.n)
    agent = PPOAgent(
        action_space=env.action_space,
        network_fn=network_fn,
        network_name="savepoints/snake.pkl",
    )
    stats = Stats("snake")

    test_episodes = 10
    training_episodes = 100
    epochs = 1000

    for _ in range(epochs):
        test(env, agent, test_episodes, stats, RENDER)
        train(env, agent, training_episodes)

    env.close()

import time
import gym
from gym import wrappers, logger
from agent.q_agent import QAgent

from gym.envs.registration import register

register(id="snake-rc-v0", entry_point="gym_rc_snake.envs:SnakeRCEnv")
from enum import Enum


class SnakeMove(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class FoodSeekerAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        last_action, snake, food = observation
        head = snake[-1]
        if head[0] < food[0]:
            return SnakeMove.RIGHT.value
        elif head[0] > food[0]:
            return SnakeMove.LEFT.value
        elif head[1] > food[1]:
            return SnakeMove.DOWN.value
        else:
            return SnakeMove.UP.value


if __name__ == "__main__":
    logger.set_level(logger.DEBUG)
    done = False
    env = gym.make("snake-rc-v0")
    outdir = "/tmp"
    env = wrappers.Monitor(env, directory=outdir, force=True)
    agent = QAgent(env.action_space, env.board_size)
    reward = 0

    while True:
        for t in range(50000):
            ob = env.reset()
            for i in range(100):
                old_ob = ob
                action = agent.random_act()
                ob, reward, done, info = env.step(action)
                agent.update_value(old_ob, action, reward, ob)
                if done:
                    env.close()
                    break

        total_reward = 0
        for t in range(5):
            ob = env.reset()
            for i in range(100):
                action = agent.act(ob, reward, done)
                ob, reward, done, info = env.step(action)
                total_reward += reward
                time.sleep(0.03)
                env.render()
                if done:
                    env.close()
                    break
        print(f"Total reward: {total_reward}")

    env.close()

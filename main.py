import time
import gym
from gym import wrappers, logger

from gym.envs.registration import register
register(
    id='snake-rc-v0',
    entry_point='gym_rc_snake.envs:SnakeRCEnv',
)
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
        print(observation)
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


if __name__ == '__main__':
    logger.set_level(logger.DEBUG)
    done = False
    env = gym.make('snake-rc-v0')
    outdir = '/tmp'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    agent = FoodSeekerAgent(env.action_space)
    reward = 0

    for i in range(100):
        ob = env.reset()
        for t in range(1000):
            action = agent.act(ob, reward, done)
            ob, reward, done, info = env.step(action)
            env.render()
            time.sleep(.05)
            if done:
                break

    env.close()

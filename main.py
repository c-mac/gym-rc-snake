import time
import gym
from gym import wrappers, logger

from gym.envs.registration import register
register(
    id='snake-rc-v0',
    entry_point='gym_rc_snake.envs:SnakeRCEnv',
)


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    logger.set_level(logger.DEBUG)
    done = False
    env = gym.make('snake-rc-v0')
    outdir = '/tmp'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    agent = RandomAgent(env.action_space)
    reward = 0

    for i in range(100):
        ob = env.reset()
        for t in range(1000):
            action = agent.act(None, reward, done)
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(.1)
            if done:
                break

    env.close()

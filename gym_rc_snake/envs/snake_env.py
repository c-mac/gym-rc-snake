import gym
import pyglet
from gym import spaces
from gym.envs.classic_control import rendering
# from gym.utils import seeding

WINDOW_W = 1000
WINDOW_H = 800


class Window(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SnakeRCEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        print('init')
        # Left, up, right, down
        self.action_space = spaces.Discrete(4)
        self.viewer = None

    def step(self, action):
        return
        # super().step(action)

    def reset(self):
        print('reset')

    def render(self, mode='human', close=False):
        width = height = 600

        if self.viewer is None:
            self.viewer = rendering.Viewer(width, height)

        square = rendering.FilledPolygon([(0,0), (0,100), (100,100), (100,0)])
        square.set_color(0, 0, 0)
        self.viewer.add_onetime(square)

        food = rendering.FilledPolygon([(10,20), (20,20), (20,10), (10,10)])
        food.set_color(1, 0, 0)
        self.viewer.add_onetime(food)
        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def close(self):
        print('close')

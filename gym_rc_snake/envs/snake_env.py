import random
from enum import Enum
import numpy as np
import gym
import pyglet
from gym import spaces
from gym.envs.classic_control import rendering
# from gym.utils import seeding

WINDOW_SIZE = 800
BOARD_SIZE = 25
SPACE_SIZE = WINDOW_SIZE / BOARD_SIZE
START_PADDING = 2

class SnakeMove(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


class Window(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SnakeRCEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        # Left, up, right, down
        self.generate_board()
        self.action_space = spaces.Discrete(4)
        self.viewer = rendering.Viewer(WINDOW_SIZE, WINDOW_SIZE)
        self.last_action = SnakeMove.RIGHT.value

    def step(self, action):
        """
        Return: observation, reward, done, info
        """
        new_head = self.new_head(action)

        if self.last_action and not self.valid_action(action):
            new_head = self.new_head(self.last_action)
        else:
            self.last_action = action

        score = 0

        if self.kills_snake(new_head, action):
            return (self.action_space, -1, True, None)

        if new_head == self.food:
            score = 1
        else:
            self.snake = self.snake[1:]

        self.snake.append(new_head)

        return (self.action_space, score, False, None)

    def valid_action(self, action):
        # Up and Down are 0 and 2, so their difference is 2
        # The same applies to left and right (1 and 3)
        return not (
            self.last_action == action or
            abs(self.last_action - action) == 2
        )

    def new_head(self, action):

        x = self.snake[-1][0]
        y = self.snake[-1][1]

        if action == SnakeMove.DOWN.value:
            y -= 1
        if action == SnakeMove.UP.value:
            y += 1
        if action == SnakeMove.RIGHT.value:
            x += 1
        if action == SnakeMove.LEFT.value:
            x -= 1

        return [x, y]

    def kills_snake(self, move, action):
        if move in self.snake:
            import pdb; pdb.set_trace()
            return True
        elif move[0] > BOARD_SIZE or move[0] < 0 or move[1] > BOARD_SIZE or move[1] < 0:
            return True
        return False


    def reset(self):
        self.generate_board()
        print("RESET")

    def generate_board(self):
        start_head = [
            random.randint(START_PADDING, BOARD_SIZE - START_PADDING),
            random.randint(START_PADDING, BOARD_SIZE - START_PADDING)
        ]
        self.snake = [
            start_head,
            # [start_head[0] + 1, start_head[1]]
        ]
        self.food = [
            random.randint(START_PADDING, BOARD_SIZE - START_PADDING),
            random.randint(START_PADDING, BOARD_SIZE - START_PADDING)
        ]


    def render(self, mode='human', close=False):
        for segment in self.snake:
            square = self.render_square(segment[0], segment[1], (0, 0, 1))
            self.viewer.add_onetime(square)

        food = self.render_square(self.food[0], self.food[1], (1, 0, 0))
        self.viewer.add_onetime(food)
        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def render_square(self, x, y, color):
        x *= SPACE_SIZE
        y *= SPACE_SIZE
        square = rendering.FilledPolygon([(x, y),
                                          (x + SPACE_SIZE, y),
                                          (x + SPACE_SIZE, y + SPACE_SIZE),
                                          (x, y + SPACE_SIZE)])
        square.set_color(*color)
        return square

    def close(self):
        print('close')

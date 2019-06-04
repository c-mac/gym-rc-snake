import random
from enum import Enum
import numpy as np
import gym
import pyglet
from gym import spaces
from gym.envs.classic_control import rendering

# from gym.utils import seeding

WINDOW_SIZE = 800
BOARD_SIZE = 12
SPACE_SIZE = WINDOW_SIZE / BOARD_SIZE
START_PADDING = 2


class SnakeMove(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


class SnakeRCEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        # Left, up, right, down
        self.generate_board()
        self.action_space = spaces.Discrete(4)
        self.viewer = rendering.Viewer(WINDOW_SIZE, WINDOW_SIZE)
        self.last_action = SnakeMove.RIGHT.value
        self.board_size = BOARD_SIZE

    def step(self, action):
        """
        Return: observation, reward, done, info
        """
        new_head = self.new_head(action)

        if self.last_action != None and not self.valid_action(action):
            new_head = self.new_head(self.last_action)
        else:
            self.last_action = action

        reward = -0.1

        self.snake.append(new_head)
        if self.snake_dead():
            return (self.observation(), self.snake_dead(), True, None)
        else:
            ob = self.observation()

            if new_head == self.food:
                self.food = [
                    random.randint(START_PADDING, BOARD_SIZE - START_PADDING),
                    random.randint(START_PADDING, BOARD_SIZE - START_PADDING),
                ]
                reward = 100
            else:
                self.snake = self.snake[1:]

            return (ob, reward, False, None)

    def valid_action(self, action):
        # Up and Down are 0 and 2, so their difference is 2
        # The same applies to left and right (1 and 3)
        return not (self.last_action == action or abs(self.last_action - action) == 2)

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

    def snake_dead(self):
        head = self.snake[-1]
        tail = self.snake[:-1]
        if head in tail:
            return -10
        elif (
            head[0] >= BOARD_SIZE or head[0] < 0 or head[1] >= BOARD_SIZE or head[1] < 0
        ):
            return -100
        return False

    def reset(self):
        self.generate_board()
        return self.observation()

    def generate_board(self):
        start_head = [
            random.randint(START_PADDING, BOARD_SIZE - 1 - START_PADDING),
            random.randint(START_PADDING, BOARD_SIZE - 1 - START_PADDING),
        ]
        self.snake = [start_head, [start_head[0] + 1, start_head[1]]]
        self.food = [
            random.randint(0, BOARD_SIZE - 1),
            random.randint(0, BOARD_SIZE - 1),
        ]

    def render(self, mode="human", close=False):
        for segment in self.snake:
            square = self.render_square(segment[0], segment[1], (0, 0, 1))
            self.viewer.add_onetime(square)

        food = self.render_square(self.food[0], self.food[1], (1, 0, 0))
        self.viewer.add_onetime(food)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def render_square(self, x, y, color):
        x *= SPACE_SIZE
        y *= SPACE_SIZE
        square = rendering.FilledPolygon(
            [
                (x, y),
                (x + SPACE_SIZE, y),
                (x + SPACE_SIZE, y + SPACE_SIZE),
                (x, y + SPACE_SIZE),
            ]
        )
        square.set_color(*color)
        return square

    def observation(self):
        return (self.last_action, self.snake, self.food)

import numpy as np
import random
import gym

from enum import IntEnum
from gym import spaces

WINDOW_SIZE = 800
BOARD_SIZE = 8
START_PADDING = 1


class Move(IntEnum):
    RIGHT = -1
    STRAIGHT = 0
    LEFT = 1


class Direction(IntEnum):
    NORTH = 0
    WEST = 1
    SOUTH = 2
    EAST = 3


# These are listed in counter-clockwise order. This means that you can "add" Moves
# to them to get a turn.  For instance, if our current direction is NORTH (or zero as
# noted below), we can turn RIGHT by adding RIGHT to our index (returning us WEST as our
# next transformation)
TRANSFORMATIONS = [
    # NORTH
    [0, 1],
    # WEST
    [-1, 0],
    # SOUTH
    [0, -1],
    # EAST
    [1, 0],
]


class SnakeRCEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 25}

    def __init__(self, board_size=BOARD_SIZE):
        self.action_space = spaces.Discrete(3)
        self.current_direction = Direction.NORTH
        self.board_size = board_size
        self.space_size = WINDOW_SIZE / BOARD_SIZE
        self.generate_board()
        self.viewer = None

    def turn(self, action):
        new_direction = self.current_direction + action

        if new_direction == -1:
            # We were going NORTH and we turned RIGHT
            new_direction = Direction.EAST
        if new_direction == 4:
            # We were going EAST and we turned LEFT
            new_direction = Direction.NORTH

        return new_direction

    def step(self, action):
        """
        Return: observation, reward, done, info
        """
        action = action - 1
        head = self.snake[-1]
        self.current_direction = self.turn(action)
        new_head = self.new_head()

        prev_distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        reward = -0.03 + 0.01 * (prev_distance - new_distance)
        done = False

        self.snake.append(new_head)
        if self.snake_dead():
            return (self.observation(), self.snake_dead(), True, None)
        else:
            ob = self.observation()

            if new_head == self.food:
                reward = 2.0
                if len(self.snake) == self.board_size ** 2:
                    done = True
                while not done and self.food in self.snake:
                    self.food = [
                        random.randint(0, self.board_size - 1),
                        random.randint(0, self.board_size - 1),
                    ]
            else:
                self.snake = self.snake[1:]

            return (ob, reward, done, None)

    def new_head(self, direction=None, head=None, step_size=1):
        if not direction:
            direction = self.current_direction
        if not head:
            head = self.snake[-1]

        transform = TRANSFORMATIONS[direction]

        return [head[0] + transform[0] * step_size, head[1] + transform[1] * step_size]

    def snake_dead(self):
        head = self.snake[-1]
        tail = self.snake[:-1]
        if head in tail:
            return -1.0
        elif (
            head[0] >= self.board_size
            or head[0] < 0
            or head[1] >= self.board_size
            or head[1] < 0
        ):
            return -1.0
        return False

    def reset(self):
        self.generate_board()
        return self.observation()

    def generate_board(self):
        self.snake = [
            [
                random.randint(START_PADDING, self.board_size - 1 - START_PADDING),
                random.randint(START_PADDING, self.board_size - 1 - START_PADDING),
            ]
        ]
        self.food = [
            random.randint(0, self.board_size - 1),
            random.randint(0, self.board_size - 1),
        ]

    def render(self, mode="human", close=False):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_SIZE, WINDOW_SIZE)

        for segment in self.snake:
            square = self.render_square(segment[0], segment[1], (0, 0, 1), rendering)
            self.viewer.add_onetime(square)

        food = self.render_square(self.food[0], self.food[1], (1, 0, 0), rendering)
        self.viewer.add_onetime(food)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def render_square(self, x, y, color, r):
        x *= self.space_size
        y *= self.space_size
        square = r.FilledPolygon(
            [
                (x, y),
                (x + self.space_size, y),
                (x + self.space_size, y + self.space_size),
                (x, y + self.space_size),
            ]
        )
        square.set_color(*color)
        return square

    def observation(self):
        return (self.current_direction, self.snake, self.food)

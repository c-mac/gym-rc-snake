import random
from enum import Enum
import gym
from gym import spaces
from gym.envs.classic_control import rendering

WINDOW_SIZE = 800
BOARD_SIZE = 8
START_PADDING = 0


class SnakeMove(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


TRANSFORMATIONS = {
    SnakeMove.DOWN.value: [0, -1],
    SnakeMove.UP.value: [0, 1],
    SnakeMove.RIGHT.value: [1, 0],
    SnakeMove.LEFT.value: [-1, 0],
}


class SnakeRCEnv(gym.Env):
    def __init__(self, board_size=BOARD_SIZE, render=False):
        self.action_space = spaces.Discrete(4)
        if render:
            self.viewer = rendering.Viewer(WINDOW_SIZE, WINDOW_SIZE)
        self.last_action = SnakeMove.RIGHT.value
        self.board_size = board_size
        self.space_size = WINDOW_SIZE / BOARD_SIZE
        self.generate_board()

    def step(self, action):
        """
        Return: observation, reward, done, info
        """

        action = action.item()
        head = self.snake[-1]
        new_head = self.new_head(action)

        if not self.valid_action(action):
            new_head = self.new_head(self.last_action)
        else:
            self.last_action = action

        prev_distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        reward = -0.02 + 0.01 * (prev_distance - new_distance)
        done = False

        self.snake.append(new_head)
        if self.snake_dead():
            return (self.observation(), self.snake_dead(), True, None)
        else:
            ob = self.observation()

            if new_head == self.food:
                reward = 1.0
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

    def valid_action(self, action):
        # Up and Down are 0 and 2, so their difference is 2
        # The same applies to left and right (1 and 3)
        return not (self.last_action == action or abs(self.last_action - action) == 2)

    def new_head(self, action):
        old_head = self.snake[-1]

        transform = TRANSFORMATIONS[action]

        return [old_head[0] + transform[0], old_head[1] + transform[1]]

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
        start_head = [
            random.randint(START_PADDING, self.board_size - 1 - START_PADDING),
            random.randint(START_PADDING, self.board_size - 1 - START_PADDING),
        ]
        self.snake = [start_head, [start_head[0] + 1, start_head[1]]]
        self.food = [
            random.randint(0, self.board_size - 1),
            random.randint(0, self.board_size - 1),
        ]

    def render(self, mode="human", close=False):
        for segment in self.snake:
            square = self.render_square(segment[0], segment[1], (0, 0, 1))
            self.viewer.add_onetime(square)

        food = self.render_square(self.food[0], self.food[1], (1, 0, 0))
        self.viewer.add_onetime(food)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def render_square(self, x, y, color):
        x *= self.space_size
        y *= self.space_size
        square = rendering.FilledPolygon(
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
        return (self.last_action, self.snake, self.food)

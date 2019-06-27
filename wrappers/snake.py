import torch
import numpy as np

from gym import ObservationWrapper, spaces
from gym_rc_snake.envs.snake_env import Move, Direction


class EntireBoard(ObservationWrapper):
    def __init__(self, env):
        super(EntireBoard, self).__init__(env)
        self.observation_space = spaces.Tuple(
            (
                # current direction
                spaces.Discrete(4),
                # board shape
                spaces.Box(
                    low=0,
                    high=2,
                    shape=(self.board_size, self.board_size),
                    dtype=np.int,
                ),
            )
        )

    def observation(self, observation):
        board = np.zeros([self.board_size, self.board_size], dtype=np.int)
        for s in self.snake:
            board[np.clip(s[0], 0, 7), np.clip(s[1], 0, 7)] = 1
        board[self.food[0], self.food[1]] = 2

        return board


class SnakePerspectiveMultipleFrames(ObservationWrapper):
    def observation(self, observation):
        if not getattr(self, "observations", None):
            self.observation_space = spaces.Box(low=-1, high=2, shape=(70,))
            self.observations = [[0] * 7] * 10

        current_direction, snake, food = observation
        head = snake[-1]
        left = self.new_head(head=head, direction=self.turn(Move.LEFT))
        left2 = self.new_head(head=head, direction=self.turn(Move.LEFT), step_size=2)
        straight = self.new_head(head=head, direction=self.turn(Move.STRAIGHT))
        straight2 = self.new_head(
            head=head, direction=self.turn(Move.STRAIGHT), step_size=2
        )
        right = self.new_head(head=head, direction=self.turn(Move.RIGHT))
        right2 = self.new_head(head=head, direction=self.turn(Move.RIGHT), step_size=2)

        def what_is_there(location):
            if location in snake[:-1]:
                return 1
            if location == food:
                return -1
            if location[0] > 7 or location[0] < 0 or location[1] > 7 or location[1] < 0:
                return 2
            else:
                return 0

        ob = [current_direction] + list(
            map(what_is_there, [left2, left, straight, straight2, right, right2])
        )

        self.observations.append(ob)
        self.observations = self.observations[-10:]

        return torch.tensor(self.observations).flatten(1)


class SnakePerspective(ObservationWrapper):
    def observation(self, observation):
        current_direction, snake, food = observation
        head = snake[-1]

        left = self.new_head(head=head, direction=self.turn(Move.LEFT))
        straight = self.new_head(head=head, direction=self.turn(Move.STRAIGHT))
        right = self.new_head(head=head, direction=self.turn(Move.RIGHT))

        def what_is_there(location):
            if location in snake[:-1]:
                return 1
            if location == food:
                return -1
            if location[0] > 7 or location[0] < 0 or location[1] > 7 or location[1] < 0:
                return 1
            else:
                return 0

        ob = list(map(what_is_there, [left, straight, right]))

        return ob


class SnakePerspectiveWithPrevActions(ObservationWrapper):
    DIRECTION_HISTORY = 8

    def __init__(self, env):
        super(SnakePerspectiveWithPrevActions, self).__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=((6 + self.DIRECTION_HISTORY,)), dtype=np.int
        )

    def observation(self, observation):
        if not getattr(self, "observations", None):
            self.directions = [0] * self.DIRECTION_HISTORY

        current_direction, snake, food = observation
        head = snake[-1]
        left = self.new_head(head=head, direction=self.turn(Move.LEFT))
        left2 = self.new_head(head=head, direction=self.turn(Move.LEFT), step_size=2)
        straight = self.new_head(head=head, direction=self.turn(Move.STRAIGHT))
        straight2 = self.new_head(
            head=head, direction=self.turn(Move.STRAIGHT), step_size=2
        )
        right = self.new_head(head=head, direction=self.turn(Move.RIGHT))
        right2 = self.new_head(head=head, direction=self.turn(Move.RIGHT), step_size=2)

        def what_is_there(location):
            if location in snake[:-1]:
                return 1
            if location == food:
                return -1
            if location[0] > 7 or location[0] < 0 or location[1] > 7 or location[1] < 0:
                return 1
            else:
                return 0

        ob = list(map(what_is_there, [left2, left, straight, straight2, right, right2]))

        self.directions.append(self.current_direction)
        self.directions = self.directions[-self.DIRECTION_HISTORY :]

        return torch.tensor(ob + self.directions)

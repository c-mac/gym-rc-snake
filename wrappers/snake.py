import numpy as np

from gym import ObservationWrapper
from gym_rc_snake.envs.snake_env import Move, Direction


class BoardOnly(ObservationWrapper):
    def observation(self, observation):
        board = np.zeros([self.board_size, self.board_size], dtype=np.int)
        for s in self.snake:
            board[np.clip(s[0], 0, 7), np.clip(s[1], 0, 7)] = 1
        board[self.food[0], self.food[1]] = 2

        return board


class SnakePerspective(ObservationWrapper):
    def observation(self, observation):
        current_direction, snake, food = observation
        left = self.new_head(self.turn(Move.LEFT))
        left2 = self.new_head(self.turn(Move.LEFT), 2)
        straight = self.new_head(self.turn(Move.STRAIGHT))
        straight2 = self.new_head(self.turn(Move.STRAIGHT), 2)
        right = self.new_head(self.turn(Move.RIGHT))
        right2 = self.new_head(self.turn(Move.RIGHT), 2)

        def what_is_there(location):
            if location in snake[:-1]:
                return 1
            if location == food:
                return -1
            if location[0] > 7 or location[0] < 0 or location[1] > 7 or location[1] < 0:
                return 2
            else:
                return 0

        return list(
            map(what_is_there, [left2, left, straight, straight2, right, right2])
        )

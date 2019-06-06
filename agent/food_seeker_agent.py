from enum import Enum


class SnakeMove(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


class FoodSeekerAgent(object):
    """Snake want food."""

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

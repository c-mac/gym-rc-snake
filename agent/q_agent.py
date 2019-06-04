import random
import numpy as np


class QAgent(object):
    """
    Take in board state and use a Q Learning algorithm to decide the next state
    """

    GAMMA = 0.9
    ALPHA = 0.2

    DEFAULT_ROW = {0: 0, 1: 0, 2: 0, 3: 0}

    def __init__(self, action_space, board_size):
        self.action_space = action_space
        self.q_table = {}
        self.board_size = board_size

    def random_act(self, ob):
        best_action, best_value = self.best_value_and_action(ob)
        if random.random() > 0.5:
            return best_action
        return self.action_space.sample()

    def observation_string(self, ob):
        _, snake, food = ob
        return f"[{snake[-1][0]},{snake[-1][1]}][{food[0]},{food[1]}]"

    def update_value(self, old_ob, action, reward, ob):
        _, next_value = self.best_value_and_action(ob)
        new_value = reward + self.GAMMA * next_value
        key = self.observation_string(old_ob)
        values_row = self.q_table.get(key, {**self.DEFAULT_ROW})
        old_value = values_row[action]
        values_row[action] = (1 - self.ALPHA) * old_value + self.ALPHA * new_value
        self.q_table[key] = values_row

    def best_value_and_action(self, ob):
        best_action = None
        best_value = 0
        value_row = self.q_table.get(self.observation_string(ob), False)
        if not value_row:
            value_row = self.DEFAULT_ROW
        for action, value in value_row.items():
            if best_action == None or value > best_value:
                best_value = value
                best_action = action

        return best_action, best_value

    def act(self, observation):
        action, _ = self.best_value_and_action(observation)

        return action

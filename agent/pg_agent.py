import tensorflow as tf
import numpy as np

def mlp(inputs_layer, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    current_layer = inputs_layer
    for size in sizes[:-1]:
        current_layer = tf.layers.dense(current_layer, units=size, activation=activation)
    return tf.layers.dense(current_layer, units=sizes[-1], activation=output_activation)

class PGAgent(object):
    """
    Take in board state and use a DNN to decide the next action
    """

    def __init__(self, action_space, board_size):
        self.action_space = action_space
        self.dnn = mlp(tf.placeholder(dtype=np.float32, shape=[None, board_size ** 2]), [200])


    def act(self, observation, reward, done):
        last_action, snake, food = observation
        return 0

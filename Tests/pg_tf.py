import gym
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

class HiddenLayer:
    def __init__(self, M1, M2 , f=tf.nn.tanh, use_bias = True, zeros = False):
        if zeros:
            W = np.zeros(((M1, M2)), dtype=np.float32)

        else:
            W = tf.random_normal(shape = (M1, M2)) * np.sqrt(2. / M1, dtype= np.float32)
        self.W = tf.variable(W)
        self.use_bas
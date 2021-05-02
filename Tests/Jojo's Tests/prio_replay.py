
import numpy as np
from collections import deque
import random
import tensorflow as tf

class PrioReplay():

    def __init__(self,bufferlen):

        self.a = 0.7
        self.b = 0.5

        self.buffer = deque(maxlen=bufferlen)
        self.priorities = deque(maxlen=bufferlen)

    def add(self, exp):

        self.buffer.append(exp)
        self.priorities.append(max(self.priorities, default=1))

    def sample(self,batch_size):

        scaled_priorities = np.array(self.priorities) ** self.a
        sample_probabilities = scaled_priorities / sum(scaled_priorities)

        importance = 1 / ((len(self.buffer) ** self.b) * (sample_probabilities ** self.b))
        importance_weights = 1 / max(importance)

        samples = random.choices(self.buffer, weights=self.priorities, k=batch_size)

        return samples, importance_weights

    def set_prio(self,error):

        self.priorities = abs(error)




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

    def get_importance(self,probabilities):

        importance = 1 / ((len(self.buffer) ** self.b) * (sample_probabilities ** self.b))
        importance_weights = 1 / max(importance)

        return importance_weights

    def sample(self,batch_size):

        scaled_priorities = np.array(self.priorities) ** self.a
        sample_probabilities = scaled_priorities / sum(scaled_priorities)

        indices = random.choices(range(len(self.buffer)), weights=sample_probabilities, k=batch_size)
        samples = np.array(self.buffer)[indices]
        importance_weights = self.get_importance(sample_probabilities[indices])

        return samples, importance_weights, indices

    def set_prio(self,errors,indices):
        for i, e in zip(indices, errors):
            self.priorities[indices] = abs(errors)



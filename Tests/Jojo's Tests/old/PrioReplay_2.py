
import numpy as np
from collections import deque
import random

class PrioReplay():

    def __init__(self,bufferlen):

        self.a = 0.7
        self.b = 0.5

        self.K = 32

        self.buffer = deque(maxlen=bufferlen)
        self.priorities = deque(maxlen=bufferlen)

    def add(self, exp):
        # exp = [state, action, reward, next_state]

        self.buffer.append(exp)

        self.priorities.append(max(self.priorities, default=1))

    def sample(self):

        for j in range(len(self.buffer)):

            sum_prio = sum(np.array(self.priorities) ** self.a)
            sample_probabilities = (self.priorities ** self.a) / sum_prio

            importance = 1 / ( (len(self.buffer) ** self.b) * (sample_probabilities ** self.b) )
            importance_weights = 1 / max(importance)

            [state, action, reward, next_state] = self.buffer[j]
            error = reward + discout * target_critic(next_state, target_actor(next_state)) - critic(state, action)

            self.priorities[j] = abs(error)

        L = 1/self.K * sum(importance_weights*self.priorities**2)

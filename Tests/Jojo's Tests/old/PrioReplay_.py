
import numpy as np
from collections import deque
import random

class PrioReplay():

    def __init__(self,maxlen):
        self.offset = 0.1
        self.a = 0.7

        self.importance = deque(maxlen=maxlen)

        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)

    def add(self, exp, error):
        prio = abs(error) + self.offset

        self.buffer.append(exp)
        self.priorities.append(prio)
        self.importance.append(0)

        prio_sum = 0
        for i in range(0, len(self.priorities)):
            prio_sum = prio_sum + self.priorities[i]

        for j in range(0, len(self.importance)):
            self.importance[j] = (self.priorities[j] ** self.a) / (prio_sum ** self.a)

    def sample(self):

        return random.choices(self.buffer, weights=self.importance, k=1)[0]

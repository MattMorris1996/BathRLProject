
import numpy as np
from collections import deque
import random

class PrioReplay():

    def __init__(self,bufferlen ,batch_size):

        self.a = 0.7
        self.b = 0.5

        self.K = batch_size

        self.buffer = deque(maxlen=bufferlen)
        self.priorities = deque(maxlen=bufferlen)

    def add(self, exp):
        # exp = [state, action, reward, next_state, terminal]

        self.buffer.append(exp)

        self.priorities.append(max(self.priorities, default=1))

    def sample(self):

        return random.choices(self.buffer, weights=self.priorities, k=self.K)[0]

    def prio_replay(self,target_actor,target_critic,critic):

        for j in range(len(self.buffer)):

            scaled_priorities = np.array(self.priorities) ** self.a
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

            importance = 1 / ( (len(self.buffer) ** self.b) * (sample_probabilities ** self.b) )
            importance_weight = 1 / max(importance)

            [state, action, reward, next_state, terminal] = self.buffer[j]

            print([state, action, reward, next_state, terminal])

            target_action = target_actor(next_state, training=True)
            q_target = reward + self.gamma * target_critic([next_state, target_action], training=True)
            q_current = critic([state, action], training=True)

            error = q_target - q_current

            self.priorities[j] = abs(error)

        loss = 1/self.K * sum(importance_weight*np.array(self.priorities)**2)

        return loss


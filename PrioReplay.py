import tensorflow as tf
from collections import deque
import random
import numpy as np

class PrioReplay():
    # self coded based on publication by:
    # Y. Hou , Y. Zhang, “Improving DDPG via Prioritized Experience Replay,” May 2019
    # unless specified otherwise

    def __init__(self, bufferlen):
        self.a = 0.7
        self.b = 0.5

        self.buffer = deque(maxlen=int(bufferlen))
        self.priorities = deque(maxlen=int(bufferlen))

    def add(self, exp):
        # inspire by:
        # https://github.com/the-computer-scientist/OpenAIGym/blob/master/PrioritizedExperienceReplayInOpenAIGym.ipynb
        self.buffer.append(exp)
        self.priorities.append(max(self.priorities, default=1))

    def get_importance(self, sample_probabilities):
        # inspire by:
        # https://github.com/the-computer-scientist/OpenAIGym/blob/master/PrioritizedExperienceReplayInOpenAIGym.ipynb
        importance = 1 / ((len(self.buffer) ** self.b) * (sample_probabilities ** self.b))
        importance_weights = importance / max(importance)

        return importance_weights

    def sample(self, batch_size, target_actor, target_critic, critic, discount):

        #prio_arr = np.array(self.priorities)
        #rank = [sorted(prio_arr)[::-1].index(x) + 1 for x in prio_arr]

        #scaled_priorities = np.power(rank, -self.a)
        scaled_priorities = np.array(self.priorities) ** self.a

        sample_probabilities = scaled_priorities / sum(scaled_priorities)

        indices = random.choices(range(len(self.buffer)), weights=sample_probabilities, k=batch_size)
        samples = np.array(list(self.buffer), dtype=object)[indices]
        samples_importance = self.get_importance(sample_probabilities[indices])

        for i in indices:

            [state, action, reward, next_state, terminal] = self.buffer[i]

            state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action = tf.expand_dims(tf.convert_to_tensor(action), 0)
            next_state = tf.expand_dims(tf.convert_to_tensor(next_state), 0)

            target_action = target_actor(next_state)
            q_target = reward + discount * tf.squeeze(target_critic([next_state, target_action])).numpy()
            q_current = tf.squeeze(critic([state, action])).numpy()
            error = q_target-q_current

            self.set_prio(error, i)

        importance_tf = tf.convert_to_tensor(samples_importance)
        importance_tf32 = tf.cast(importance_tf, dtype=tf.float32)
        importance_tf32_t = tf.reshape(importance_tf32, [64, 1])

        return samples, importance_tf32_t

    def set_prio(self, error, indice):
        # inspire by:
        # https://github.com/the-computer-scientist/OpenAIGym/blob/master/PrioritizedExperienceReplayInOpenAIGym.ipynb
        self.priorities[indice] = abs(error)

    def grow_b(self):

        if self.b < 1:
            self.b = self.b + 0.5/30
# DQN Implementation

import gym

import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
# from keras.optimizers import SGD

import matplotlib.pyplot as plt

from collections import deque

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

NUM_PARALLEL_EXEC_UNITS = 12

#Assume that the number of cores per socket in the machine is denoted as NUM_PARALLEL_EXEC_UNITS
#  when NUM_PARALLEL_EXEC_UNITS=0 the system chooses appropriate settings

config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                        inter_op_parallelism_threads=4,
                        allow_soft_placement=False,
                        device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})

session = tf.Session(config=config)

seed = 1
num_episodes = 15000
max_steps = 250
exploring_starts = 20


class Agent:

    epsilon = 1
    gamma = 0.95
    alpha = 0.05
    tau = 0.1
    decay = 0.995  # 9999
    decay_a = 0.9
    noise = [0]  # np.random.normal(0, 0.1, 10000)
    memory = deque(maxlen=int(1e6))
    learn_start = 5000

    def __init__(self, env, seed):

        self.env = env
        random.seed(seed)

        self.model = self.createModel()
        self.target_model = self.createModel()

        self.state_action_model = self.createModel()

        self.reset()

    def createModel(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.alpha))
        # model.compile(loss="mean_squared_error", optimizer=SGD(lr=self.alpha))
        return model

    def replayBuffer(self, state, action, reward, new_state, terminal):
        self.memory.append([state, action, reward, new_state, terminal])

    def replay(self):
        batch_size = 32
        state_update = np.zeros([batch_size, 2])
        target_update = np.zeros([batch_size, 3])
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for idx, sample in enumerate(samples):
            state, action, reward, new_state, terminal = sample
            target = self.target_model.predict(state)
            action = int(action)
            # print(target)
            if terminal:
                target[0][action] = reward
                # target[0] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
                # target[0] = reward + Q_future * self.gamma
            state_update[idx] = state
            target_update[idx] = target[0]
        self.model.fit(state_update, target_update, epochs=5, verbose=0)

    def trainTarget(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def train(self, state, action, reward, next_state, terminal, steps):
        self.replayBuffer(state, action, reward, next_state, terminal)
        if steps % 25 == 0 and len(self.memory) > self.learn_start:
            self.replay()
        if steps % 200 == 0 and len(self.memory) > self.learn_start:
            self.trainTarget()

    def reset(self):
        self.epsilon *= self.decay
        self.alpha *= self.decay_a
        self.epsilon = max(self.epsilon / 800, self.epsilon)
        self.alpha = max(self.decay / 600, self.decay)
        # K.set_value(self.model.optimizer.learning_rate, self.alpha)
        # K.set_value(self.target_model.optimizer.learning_rate, self.alpha)
        # K.set_value(self.state_action_model.optimizer.learning_rate, self.alpha)
        # print(self.epsilon)
        # pass

    def chooseAction(self, state):
        # self.epsilon *= self.decay
        self.epsilon = max(self.epsilon/1000, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # else:
        # print((self.model.predict(state)[0]))
            # action = action_us*2 - 1
        return np.argmax(self.model.predict(state)[0]) + random.choice(self.noise) # action self.model.predict(state)  #


def main():
    env = gym.make('MountainCar-v0').env #Continuous-v0').env
    agent = Agent(env, seed)
    rewards = np.zeros(num_episodes)
    rewards_100 = deque(maxlen=int(100))
    averages = np.ones(num_episodes)*(-max_steps)
    plt.ion()
    for episode in range(num_episodes):
        action = np.zeros(1)
        state = env.reset().reshape(1, 2)
        total_reward = 0
        for step in range(max_steps):
            # env.render()
            action = agent.chooseAction(state)
            # print(state, action)
            # print(step)
            # print(total_reward)
            next_state, reward, terminal, info = env.step(int(action))  # env.action_space.sample())  # take a random action
            next_state = next_state.reshape(1, 2)
            # if state[0][0] > -0.5:
            #     reward += (state[0][0]+1) * 0.1  # next_state[0][0] -
            state = next_state
            if state[0][0] >= 0.5:
                reward += 100
            agent.train(state, action, reward, next_state, terminal, step)
            total_reward += reward
            if step % 50 == 0 and episode >= exploring_starts:
                agent.reset()
            if terminal:
                # print(info)
                agent.trainTarget()
                break
        env.close()
        # print(total_reward)
        rewards[episode] = total_reward
        rewards_100.append(total_reward)
        averages[episode] = np.mean(rewards_100)
        if episode >= exploring_starts:
            #agent.reset()
            # agent.trainTarget()
            pass
        if abs(np.mean(rewards_100)) >= 10:  # step >= 199:
            print("Failed to complete in episode {} with reward of {} in {} steps, average reward of last 100 episodes is {}".format(episode, total_reward, step+1, np.mean(rewards_100)))

        else:
            print("Completed in {} episodes, with reward of {}, average reward of {}".format(episode, total_reward, np.mean(rewards_100)))
            # terminal = False
            # state = env.reset().reshape(1, 2)
            # while not terminal:
            #     env.render()
            #     action = agent.chooseAction(state)
            #     next_state, reward, terminal, info = env.step(action)  # env.action_space.sample())  # take a random action
            #     next_state = next_state.reshape(1, 2)
            #     state = next_state

            break
        plt.plot(averages)
        plt.draw()
        plt.pause(0.00001)
        plt.clf()
    # env.close()


if __name__ == "__main__":
    main()

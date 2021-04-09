import gym

import random
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# from keras.optimizers import SGD

import matplotlib.pyplot as plt

from collections import deque

# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

NUM_PARALLEL_EXEC_UNITS = 12

# Assume that the number of cores per socket in the machine is denoted as NUM_PARALLEL_EXEC_UNITS
#  when NUM_PARALLEL_EXEC_UNITS=0 the system chooses appropriate settings

config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                        inter_op_parallelism_threads=4,
                        allow_soft_placement=True,
                        device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})

session = tf.Session(config=config)

seed = 1
num_episodes = 3000
max_steps = 1500
exploring_starts = 4

render_list = []#0, 10, 20, 30, 31, 32, 33, 34, 35]

class Agent:
    epsilon = 1
    learn_start = 5000
    gamma = 0.99
    alpha = 0.01
    # alpha2 = 0.005
    tau = 0.1
    decay = 0.995  # 9999
    noise = [0] #np.random.normal(0, 0.2, 1000)
    memory = deque(maxlen=int(1e6))

    def __init__(self, env, seed):

        self.env = env
        random.seed(seed)
        self.env.seed(seed)
        self.model = self.createModel()
        self.target_model = self.createModel()

        self.state_action_model = self.createModel(self.env.observation_space.shape[0] + 1)
        self.state_action_target_model = self.createModel(self.env.observation_space.shape[0] + 1)

        self.reset()

    def createModel(self, input=None):
        model = Sequential()
        # state_shape = self.env.observation_space.shape
        if input is None:
            input = self.env.observation_space.shape[0]
        # print(input)
        model.add(Dense(256, input_dim=input, activation="relu")) #24
        model.add(Dense(256, activation="relu")) #48
        model.add(Dense(256, activation="relu")) #24
        # self.env.action_space.n))
        if input is self.env.observation_space.shape[0]:
            model.add(Dense(1, activation="tanh"))
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.alpha/100,
                decay_steps=10000,
                decay_rate=1)
            model.compile(loss="huber_loss", optimizer=Adam(learning_rate=lr_schedule))
        else:
            model.add(Dense(1))
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.alpha/10,
                decay_steps=10000,
                decay_rate=1)
            model.compile(loss="huber_loss", optimizer=Adam(learning_rate=lr_schedule)) #mean_squared_error
        # model.compile(loss="mean_squared_error", optimizer=SGD(lr=self.alpha))
        return model

    def replayBuffer(self, state, action, reward, next_state, terminal):
        self.memory.append([state, action, reward, next_state, terminal])

    def replay(self):
        batch_size = 32
        state_update = np.zeros([batch_size, 2])
        target_update = np.zeros(batch_size)
        Q_expected = np.zeros([batch_size, 3])
        Q_expected_temp = np.zeros([batch_size, 3])
        Q_target = np.zeros(batch_size)
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for idx, sample in enumerate(samples):
            state, action, reward, next_state, terminal = sample
            next_action = self.target_model.predict(next_state)
            # print(next_action)
            Q_expected[idx][0] = next_state[0][0]
            Q_expected[idx][1] = next_state[0][1]
            Q_expected[idx][2] = next_action[0]
            action_pred = self.model.predict(state)
            Q_expected_temp[idx][0] = state[0][0]
            Q_expected_temp[idx][1] = state[0][1]
            Q_expected_temp[idx][2] = action[0]
            state_update[idx] = state
            # print(Q_expected[idx].reshape(1, 3)[0])
            Q_target_next = self.state_action_target_model.predict(Q_expected[idx].reshape(1, 3))
            # Q_current = self.state_action_target_model.predict(Q_expected_temp[idx].reshape(1, 3))
            # print(reward)
            Q_target[idx] = ((reward + (self.gamma * Q_target_next[0] ))* (1 - terminal)) #- Q_current
            # Q_expected[idx] = self.state_action_model(state, action)
            Q_expected_temp[idx][2] = action_pred[0]

            target_update[idx] = (self.state_action_model.predict(Q_expected_temp[idx].reshape(1, 3)))#+action
            # print(self.state_action_model.predict(Q_expected_temp[idx].reshape(1, 3)))
            # Q_expected[idx][0] = state[0][0]
            # Q_expected[idx][1] = state[0][1]
            # Q_expected[idx][2] = action[0]
            # print(target_update[idx], action_pred[0][0], Q_target[idx], Q_current[0][0])
        self.model.fit(state_update, target_update, epochs=1, verbose=0)  # actor
        self.state_action_model.fit(Q_expected, Q_target, epochs=1, verbose=0)  # critic

        #     target = self.target_model.predict(state)
        #     # print(target)
        #     if terminal:
        #         # target[0][action] = reward
        #         target[0] = reward
        #     else:
        #         Q_future = max(self.target_model.predict(new_state)[0])
        #         # target[0][action] = reward + Q_future * self.gamma
        #         target[0] = reward + Q_future * self.gamma
        #     state_update[idx] = state
        #     target_update[idx] = target[0]  # [action]
        # self.model.fit(state_update, target_update, epochs=1, verbose=0)

    def trainTarget(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
        del weights, target_weights
        weights = self.state_action_model.get_weights()
        target_weights = self.state_action_target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.state_action_target_model.set_weights(target_weights)
        del weights, target_weights

    def train(self, state, action, reward, next_state, terminal, steps):
        self.replayBuffer(state, action, reward, next_state, terminal)
        if steps % 25 == 0 and len(self.memory) > self.learn_start:
            self.replay()
        if steps % 50 == 0:
            self.trainTarget()

    def reset(self):
        self.epsilon *= self.decay
        self.epsilon = max(self.epsilon / 1000, self.epsilon)
        # pass

    def chooseAction(self, state):
        # self.epsilon *= self.decay
        self.epsilon = max(self.epsilon / 1000, self.epsilon)
        if np.random.random() < self.epsilon:
            return random.uniform(-1, 1)
        # else:
        # print((self.model.predict(state)))
        # action = action_us*2 - 1
        return np.clip(self.model.predict(state) + random.choice(self.noise), -1, 1)  # np.argmax(self.model.predict(state))  # action


def main():
    env = gym.make('MountainCarContinuous-v0').env
    agent = Agent(env, seed)
    rewards = np.zeros(num_episodes)
    rewards_100 = deque(maxlen=int(100))
    averages = np.ones(num_episodes)# * (-max_steps)
    plt.ion()
    for episode in range(num_episodes):
        action = np.zeros(1)
        state = env.reset().reshape(1, 2)
        total_reward = 0
        for step in range(max_steps):
            if episode in render_list:
                env.render()
                pass
            action[0] = agent.chooseAction(state)
            # print(action)
            # reward = 0
            # print(state, action)
            # print(step)
            # print(total_reward)
            next_state, reward, terminal, info = env.step(-action)  # env.action_space.sample())  # take a random action
            next_state = next_state.reshape(1, 2)
            # reward += (state[0][0]+0.4)*(0.0001)
            # print(reward)
            agent.train(state, action, reward, next_state, terminal, step)
            state = next_state
            if state[0][0] >= 0.5:
                # reward += 100
                pass
            total_reward += reward
            if step % 250 == 0 and episode >= exploring_starts:
                agent.reset()
            if terminal:
                # print(info)
                break
        # env.close()
        # print(total_reward)
        rewards[episode:] = total_reward
        rewards_100.append(total_reward)
        averages[episode:] = np.mean(rewards_100)
        # averages[episode:]
        if episode >= exploring_starts:
            # agent.reset()
            # agent.trainTarget()
            pass
        if np.mean(rewards_100) <= 100:  # step >= 199:
            print(
                "Failed to complete in episode {} with reward of {} in {} steps, average reward of last 100 episodes "
                "is {}".format(episode, total_reward, step + 1, np.mean(rewards_100)))

        else:
            print("Completed in {} episodes, with reward of {}, average reward of {}".format(episode, total_reward,
                                                                                             np.mean(rewards_100)))
            # terminal = False
            # state = env.reset().reshape(1, 2)
            # while not terminal:
            #     env.render()
            #     action = agent.chooseAction(state)
            #     next_state, reward, terminal, info = env.step(action)  # env.action_space.sample())  # take a random action
            #     next_state = next_state.reshape(1, 2)
            #     state = next_state

            break
        plt.subplot(2, 1, 1)
        plt.plot(averages)
        plt.subplot(2, 1, 2)
        plt.plot(rewards)
        plt.draw()
        plt.pause(0.00001)
        plt.clf()
    env.close()


if __name__ == "__main__":
    main()

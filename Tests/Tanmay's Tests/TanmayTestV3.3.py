import gym

import random
import numpy as np
import pickle
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# from keras.optimizers import SGD

# import msvcrt

from Noise import OUNoise

import matplotlib.pyplot as plt

from collections import deque

# import tensorflow as tf
#
# configuration = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
# session = tf.compat.v1.Session(config=configuration)

from tensorflow.python.client import device_lib
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


plot = True

seed = 1
num_episodes = 1000
max_steps = 10000
min_steps = 1500
exploring_starts = 5
average_of = 25

step_decay = 0.995
augment = 0.01

render_list = []#0, 10, 20, 30, 40, 50, 100, 110, 120, 130, 140, 150 ]  # 50, 51, 52, 53, 100, 101, 102, 103, 104, 105] #0, 10, 20, 30, 31, 32, 33, 34, 35]


class Agent:
    epsilon = 1
    learn_start = 5000
    gamma = 0.99
    alpha = 0.01  # 0.005
    # alpha2 = 0.005
    tau = 0.001
    decay = 0.999  # 995
    # noise = OUNoise(1, seed, mu=0, theta=0.15, sigma=0.2)  # [0] #np.random.normal(0, 0.2, 1000)
    mem_len = 4.5e4  # 1.5e4
    memory = deque(maxlen=int(mem_len))

    def __init__(self, env, seed):

        self.env = env
        random.seed(seed)
        self.env.seed(seed)
        self.model = self.createModel()
        self.target_model = self.createModel()
        self.noise = OUNoise(self.env.action_space.shape[0], seed, theta=0.08,
                             sigma=0.125)  # [0] #np.random.normal(0, 0.2, 1000)
        self.state_action_model = self.createModel(self.env.observation_space.shape[0] + self.env.action_space.shape[0])
        self.state_action_target_model = self.createModel(
            self.env.observation_space.shape[0] + self.env.action_space.shape[0])

        self.reset()

    def createModel(self, input=None):
        model = Sequential()
        # state_shape = self.env.observation_space.shape
        if input is None:
            input = self.env.observation_space.shape[0]
        # print(input)
        # model.add(Dense(256, input_dim=input, activation="relu"))  # 24
        # model.add(Dense(512, activation="relu"))  # 48
        # model.add(Dense(256, activation="relu"))  # 24
        # self.env.action_space.n))
        if input is self.env.observation_space.shape[0]:  # Actor
            model.add(Dense(12, input_dim=input, activation="linear"))  # 24
            model.add(Dense(24, activation="relu"))  # 48
            model.add(Dense(12, activation="relu"))  # 24
            model.add(Dense(1, activation="tanh"))
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.alpha / 100,
                decay_steps=10000,
                decay_rate=1)
            model.compile(loss="huber_loss", optimizer=Adam(learning_rate=lr_schedule))
        else:  # Critic
            model.add(Dense(12, input_dim=input, activation="linear"))  # 24
            model.add(Dense(24, activation="linear"))  # 48
            model.add(Dense(12, activation="linear"))  # 24
            model.add(Dense(1))
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.alpha / 10,
                decay_steps=10000,
                decay_rate=1)
            model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=lr_schedule))  # mean_squared_error
        # model.compile(loss="mean_squared_error", optimizer=SGD(lr=self.alpha))
        return model

    def replayBuffer(self, state, action, reward, next_state, terminal):
        self.memory.append([state, action, reward, next_state, terminal])

    def replay(self):
        batch_size = 32  # 32
        state_update = np.zeros([batch_size, self.env.observation_space.shape[0]])
        target_update = np.zeros(batch_size)
        Q_expected = np.zeros([batch_size, self.env.observation_space.shape[0] + self.env.action_space.shape[0]])
        Q_expected_temp = np.zeros([batch_size, self.env.observation_space.shape[0] + self.env.action_space.shape[0]])
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
            Q_target_next = self.state_action_target_model.predict(Q_expected[idx].reshape(self.env.action_space.shape[0], self.env.observation_space.shape[0] + self.env.action_space.shape[0]))
            # Q_current = self.state_action_target_model.predict(Q_expected_temp[idx].reshape(1, 3))
            # print(reward)
            Q_target[idx] = (reward + (self.gamma * Q_target_next[0]) * (1 - terminal))  # - Q_current
            # Q_expected[idx] = self.state_action_model(state, action)
            Q_expected_temp[idx][2] = action_pred[0]

            target_update[idx] = (self.state_action_model.predict(Q_expected_temp[idx].reshape(self.env.action_space.shape[0], self.env.observation_space.shape[0] + self.env.action_space.shape[0])))  # +action
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
        return np.clip(1*self.model.predict(state) + self.noise.sample(), -1,
                       1)  # np.argmax(self.model.predict(state))  # action


class DataStore:
    def __init__(self, averages, rewards):
        self.averages = averages
        self.rewards = rewards


def main(max_steps):
    if plot:
        try:
            with open('data.pk1', 'rb') as qt:
                data = pickle.load(qt)
        except:
            pass

    env = gym.make('MountainCarContinuous-v0').env
    # env = gym.make('Pendulum-v0').env

    try:
        with open('agent.pk1', 'rb') as qt:
            agent = pickle.load(qt)
            agent.env = env
    except:
        agent = Agent(env, seed)
    rewards = np.zeros(num_episodes)
    rewards_av = deque(maxlen=int(average_of))
    averages = np.ones(num_episodes)  # * (-max_steps)
    # plt.ion()
    input_val = 0
    # fig = plt.figure()
    # ax1 = fig.add_subplot(2,1,1)
    # ax2 = fig.add_subplot(2,1,2)
    # line1, = ax1.plot([], lw=3)
    # line2, = ax2.plot([], lw=3)
    # fig.canvas.draw()
    # plt.show(block=False)

    for episode in range(num_episodes):
        action = np.zeros(1)
        state = env.reset().reshape(env.action_space.shape[0], env.observation_space.shape[0])#1, 2)
        total_reward = 0
        for step in range(max_steps):
            # if msvcrt.kbhit():
            #     input_val = True
            # elif step < 5 and input_val:
            #     input_val = False
            if episode in render_list or input_val:
                env.render()
                pass
            action[0] = agent.chooseAction(state)
            # print(action)
            # reward = 0
            # print(state, action)
            # print(step)
            # print(total_reward)
            next_state, reward, terminal, info = env.step(action)  # env.action_space.sample())  # take a random action
            next_state = next_state.reshape(env.action_space.shape[0], env.observation_space.shape[0])
            total_reward += reward
            reward += ((state[0][0]+1.2)**2)*(augment)
            # print(reward)
            agent.train(state, action, reward, next_state, terminal, step)
            state = next_state
            # if state[0][0] >= 0.5:
            #     # reward += 100
            #     pass
            if step % 250 == 0 and episode >= exploring_starts:
                agent.reset()
            if terminal:
                # print(info)
                agent.noise.reset()
                break
        # env.close()
        # print(total_reward)
        rewards[episode:] = total_reward
        rewards_av.append(total_reward)
        averages[episode:] = np.mean(rewards_av)
        # averages[episode:]
        if episode >= exploring_starts:
            # agent.reset()
            # agent.trainTarget()
            pass
        if np.mean(rewards_av) <= 90:  # step >= 199:
            print(
                "Failed to complete in episode {} with reward of {} in {} steps, average reward of last {} episodes "
                "is {}".format(episode, total_reward, step + 1, average_of, np.mean(rewards_av)))

        else:
            print("Completed in {} episodes, with reward of {}, average reward of {}".format(episode, total_reward,
                                                                                             np.mean(rewards_av)))
            # terminal = False
            # state = env.reset().reshape(1, 2)
            # while not terminal:
            #     env.render()
            #     action = agent.chooseAction(state)
            #     next_state, reward, terminal, info = env.step(action)  # env.action_space.sample())  # take a random action
            #     next_state = next_state.reshape(1, 2)
            #     state = next_state

            break

        # # fig, ax = plt.subplots(2)
        # plt.subplot(2, 1, 1)
        # plt.plot(averages)
        # plt.subplot(2, 1, 2)
        # plt.plot(rewards)
        # # plt.draw()
        # # gcf().canvas.flush_events()
        # plt.pause(0.0001)
        # plt.clf()

        # ax1.plot(averages)
        # ax2.plot(rewards)

        # line1.set_data(len(averages), averages)
        # line2.set_data(len(rewards), rewards)

        # fig.canvas.draw()
        # fig.canvas.flush_events()

        agent.noise.reset()

        max_steps = int(max(min_steps, max_steps*step_decay))

        if episode % 50 == 0:
        #     with open('agent.pk1', 'wb') as handle:
        #         pickle.dump(agent, handle, pickle.HIGHEST_PROTOCOL)
            data = DataStore(averages, rewards)
            with open('data_14.pk1', 'wb') as handle:
                pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

        env.close()

    # with open('agent.pk1', 'wb') as handle:
    #     pickle.dump(agent, handle, pickle.HIGHEST_PROTOCOL)
    with open('data_14.pk1', 'wb') as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # print(tf.__version__)
    # print(device_lib.list_local_devices())
    main(max_steps)

import gym

import random
import numpy as np
import pickle
from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense
from keras.optimizers import Adam

import os
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw

from Noise import OUNoise

import matplotlib.pyplot as plt

from collections import deque


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


live_plot = False
save = True

seed = 1
num_episodes = 10000
max_steps = 1000
min_steps = 1000
exploring_starts = 1
average_of = 25

step_decay = 1  # 0.995
augment = 0  # 0.001

render_list = [0, 25, 50, 100, 120, 150]#0, 10, 20, 30, 40, 50, 100, 110, 120, 130, 140, 150 ]  # 50, 51, 52, 53, 100, 101, 102, 103, 104, 105] #0, 10, 20, 30, 31, 32, 33, 34, 35]

class Actor(keras.Model):
    @tf.function
    def train_step(self, data):
        critic_value = data

        with tf.GradientTape() as tape:
        # actions = self.model(states, training=True)
        # critic_value = self.state_action_model(states_actions, training=True)
        #     critic_value = self.state_action_model(states_actions)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grad, self.trainable_variables))

class Critic(keras.Model):
    @tf.function
    def train_step(self, data):

        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(data, y_pred, regularization_losses=self.losses)

        critic_grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(critic_grad, self.trainable_variables))


class Agent:
    epsilon = 0#1
    epsilon_min = 0#0.1
    learn_start = 100
    gamma = 0.99
    alpha = 0.001  # 0.005
    tau = 0.005
    decay = 0.99  # 995
    mem_len = 1e6  # 4.5e5  # 1.5e4

    memory = deque(maxlen=int(mem_len))

    def __init__(self, env, seed):

        self.env = env
        random.seed(seed)
        np.random.seed(seed)
        # tf.random.set_seed(seed)
        self.env.seed(seed)
        self.model = self.createModel()
        self.target_model = self.createModel()
        self.noise = OUNoise(self.env.action_space.shape[0], seed, theta=0.2,
                             sigma=0.5)  # [0] #np.random.normal(0, 0.2, 1000)
        self.state_action_model = self.createModel(self.env.observation_space.shape[0] + self.env.action_space.shape[0])
        self.state_action_target_model = self.createModel(
            self.env.observation_space.shape[0] + self.env.action_space.shape[0])

        self.reset()

    def createModel(self, input=None):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        if input is None:
            input = self.env.observation_space.shape[0] # Actor
            inputs = keras.Input(shape=input)
            hidden = keras.layers.Dense(12, activation="relu")(inputs)
            hidden = keras.layers.Dense(24, activation="relu")(hidden)
            outputs = keras.layers.Dense(1, activation="tanh", kernel_initializer=last_init)(hidden)
            model = Actor(inputs, outputs)
            # model.add(Dense(256, input_dim=input, activation="linear"))  # 24
            # model.add(Dense(512, activation="relu"))  # 48
            # model.add(Dense(256, activation="linear"))  # 24
            # model.add(Dense(1, activation="tanh", kernel_initializer=last_init))
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.alpha / 10,
                decay_steps=1e9,
                decay_rate=0.5)
            model.compile(loss="huber_loss", optimizer=Adam(learning_rate=lr_schedule))
        else:  # Critic
            inputs = keras.Input(shape=input)
            hidden = keras.layers.Dense(12, activation="relu")(inputs)
            hidden = keras.layers.Dense(24, activation="relu")(hidden)
            outputs = keras.layers.Dense(1, activation="linear", kernel_initializer=last_init)(hidden)
            # model = Sequential()
            # model.add(Dense(12, input_dim=input, activation="linear"))  # 24
            # model.add(Dense(24, activation="relu"))  # 48
            # model.add(Dense(12, activation="linear"))  # 24
            # model.add(Dense(1))
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.alpha / 1,
                decay_steps=1e9,
                decay_rate=0.5)
            model = Critic(inputs, outputs)
            model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=lr_schedule))  # mean_squared_error
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
        states_actions = []
        states = []
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
            Q_target_next = self.state_action_target_model.predict(Q_expected[idx].reshape(self.env.action_space.shape[0], self.env.observation_space.shape[0] + self.env.action_space.shape[0]))
            Q_target[idx] = (reward + (self.gamma * Q_target_next[0]) * (1 - terminal))  # - Q_current
            Q_expected_temp[idx][2] = action_pred[0]
            target_update[idx] = (self.state_action_model.predict(Q_expected_temp[idx].reshape(self.env.action_space.shape[0], self.env.observation_space.shape[0] + self.env.action_space.shape[0])))  # +action
            # states_actions.append(Q_expected_temp[idx].reshape(self.env.action_space.shape[0], self.env.observation_space.shape[0] + self.env.action_space.shape[0]))
            # states.append(state.reshape(self.env.observation_space.shape[0]))

        self.model.fit(state_update, target_update, epochs=1, verbose=0)  # actor
        self.state_action_model.fit(Q_expected, Q_target, epochs=1, verbose=0)  # critic

        # with tf.GradientTape() as tape:
        #     #actions = self.model(states, training=True)
        #     # critic_value = self.state_action_model(states_actions, training=True)
        #     critic_value = self.state_action_model(states_actions)
        #     actor_loss = -tf.math.reduce_mean(critic_value)
        #
        # actor_grad = tape.gradient(actor_loss, self.model.trainable_variables)
        # self.model.optimizer.apply_gradients(zip(actor_grad, self.model.trainable_variables))
        #
        # del states, states_actions

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
        if steps % 5 == 0 and len(self.memory) > self.learn_start:
            self.replay()
        if steps % 25 == 0:
            self.trainTarget()

    def reset(self):
        self.epsilon *= self.decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def chooseAction(self, state):
        # self.epsilon *= self.decay
        self.epsilon = max(self.epsilon / 1000, self.epsilon)
        if np.random.random() < self.epsilon:
            return random.uniform(-1, 1)
        return np.clip(1*self.model.predict(state) + self.noise.sample(), -1, 1)  # np.argmax(self.model.predict(state))  # action


class DataStore:
    def __init__(self, averages, rewards):
        self.averages = averages
        self.rewards = rewards


def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num}', fill=text_color)

    return im

def main(max_steps):
    env = gym.make('MountainCarContinuous-v0').env
    # env = gym.make('Pendulum-v0').env
    agent = Agent(env, seed)
    rewards = np.zeros(num_episodes)
    rewards_av = deque(maxlen=int(average_of))
    averages = np.ones(num_episodes)  # * (-max_steps)
    # plt.ion()

    for episode in range(num_episodes):
        action = np.zeros(1)
        state = env.reset().reshape(env.action_space.shape[0], env.observation_space.shape[0])#1, 2)
        total_reward = 0
        frames = []
        for step in range(max_steps):
            if (episode in render_list) and save:
                frame = env.render(mode='rgb_array')
                frames.append(_label_with_episode_number(frame, episode_num=episode))
            elif episode in render_list:
                env.render()
                pass
            action[0] = agent.chooseAction(state)
            next_state, reward, terminal, info = env.step(action)  # env.action_space.sample())  # take a random action
            next_state = next_state.reshape(env.action_space.shape[0], env.observation_space.shape[0])
            total_reward += reward
            reward += ((state[0][0]+1.2)**2)*(augment)
            agent.train(state, action, reward, next_state, terminal, step)
            state = next_state
            if step % 250 == 0 and episode >= exploring_starts:
                agent.reset()
            if terminal:
                # agent.noise.reset()
                break
        rewards[episode:] = total_reward
        rewards_av.append(total_reward)
        averages[episode:] = np.mean(rewards_av)
        if np.mean(rewards_av) <= 90:  # step >= 199:
            print(
                "Failed to complete in episode {:4} with reward of {:8.3f} in {:5} steps, average reward of last {:4} episodes "
                "is {:8.3f}".format(episode, total_reward, step + 1, average_of, np.mean(rewards_av)))

        else:
            print("Completed in {:4} episodes, with reward of {:8.3f}, average reward of {:8.3f}".format(episode, total_reward, np.mean(rewards_av)))

            break

        if live_plot:
            plt.subplot(2, 1, 1)
            plt.plot(averages)
            plt.subplot(2, 1, 2)
            plt.plot(rewards)
            plt.pause(0.0001)
            plt.clf()

        agent.noise.reset()

        max_steps = int(max(min_steps, max_steps*step_decay))

        if episode % 50 == 0:
            data = DataStore(averages, rewards)
            with open('data_ddpg.pk1', 'wb') as handle:
                pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

        env.close()
        if frames:
            imageio.mimwrite(os.path.join('./videos/', 'agent_ep_{}.gif'.format(episode)), frames, fps=60)
            del frames

    with open('data_ddpg.pk1', 'wb') as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(max_steps)

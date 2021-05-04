import gym

import random
import os
import numpy as np
import copy
import pickle
import tensorflow.keras as keras
from keras.optimizers import Adam
from keras.layers import Dense
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

import os
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw

import matplotlib.pyplot as plt

from collections import deque
import tensorflow as tf

seed = 16
env = gym.make('MountainCarContinuous-v0')

# https://towardsdatascience.com/deep-deterministic-and-twin-delayed-deep-deterministic-policy-gradient-with-tensorflow-2-x-43517b0e0185
# https://github.com/abhisheksuran/Reinforcement_Learning/blob/master/td3withtau.ipynb

class Noise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2, open_ai=True):
        """Initialize parameters and noise process."""
        self.open_ai = open_ai
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = seed
        random.seed(seed)
        self.baseline_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(size), sigma=sigma * np.ones(size))
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        # self.decay()
        self.state = copy.copy(self.mu)
        self.baseline_noise.reset()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        if self.open_ai:
            return self.baseline_noise()
        return self.state

    def decay(self):
        self.sigma = max(0.35, self.sigma * 0.99)
        self.theta = max(0.15, self.theta * 0.995)


class ReplayMem():
    def __init__(self, max_mem, state_dim, n_actions):
        self.mem_ptr = 0
        self.mem_size = max_mem
        self.state_mem = np.zeros((max_mem, *state_dim), dtype=np.float32)
        self.action_mem = np.zeros((max_mem, n_actions), dtype=np.float32)
        self.reward_mem = np.zeros((max_mem,), dtype=np.float32)
        self.next_state_mem = np.zeros((max_mem, *state_dim), dtype=np.float32)
        self.done_memory = np.zeros((max_mem,), dtype=np.bool_)

    def store(self, state, action, reward, next_state, terminal):
        idx = self.mem_ptr % self.mem_size
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.terminal_memory[idx] = 1 - int(terminal)
        self.idx += 1

    def sample(self, batch_size):
        max_sample = min(self.mem_ptr, self.mem_size)
        batch = np.random.choice(max_sample, batch_size, replace=False)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        new_states = self.next_state_mem[batch]
        terminals = self.terminal_memory[batch]

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        rewards = tf.cast(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)

        return states, actions, rewards, new_states, terminals


class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.f1 = Dense(256, activation='relu')
        self.f2 = Dense(256, activation='relu')
        self.v = Dense(1, activation=None)

    def call(self, instate, action):
        x = self.f1(tf.concat([instate, action]), axis=1)
        x = self.f2(x)
        x = self.v(x)
        return x


class Actor(tf.keras.Model):
    def __init__(self, num_actions):
        super(Actor, self).__init__()
        self.f1 = Dense(256, activation='relu')
        self.f2 = Dense(256, activation='relu')
        self.mu = Dense(num_actions, activation='tanh')

    def call(self, state):
        x = self.f1(state)
        x = self.f2(x)
        x = self.mu(x)
        return x


class TD3Agent():
    def __init__(self, num_act=len(env.action_space.high)):
        self.actor = Actor(num_act)
        self.target_actor = Actor(num_act)
        self.crit1 = Critic()
        self.crit2 = Critic()
        self.target_crit1 = Critic()
        self.target_crit2 = Critic()
        self.batch_size = 64
        self.num_act = num_act
        self.a_opt = Adam(0.001)
        self.c_opt1 = Adam(0.002)
        self.c_opt2 = Adam(0.002)
        self.memory = ReplayMem(100000, env.observation_space.shape, num_act)
        self.noise = Noise(env.action_space.shape[0], seed, theta=0.2,
                           sigma=0.5)
        self.train_step = 0
        self.gamma = 0.99
        self.tau = 0.005
        self.min_action = env.action_space.low[0]
        self.max_action = env.action_space.high[0]
        self.actor_update_steps = 2
        self.warmup = 250
        self.target_actor.compile(optimizer=self.a_opt)
        self.target_crit1.compile(optimizer=self.c_opt1)
        self.target_crit1.compile(optimizer=self.c_opt2)

    def choose_act(self, state, evaluate=False):
        if self.train_step > self.warmup:
            evaluate = True
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += self.noise.sample()
        actions = self.max_action * (tf.clip_by_value(actions, self.min_action, self.max_action))
        return actions[0]

    def save_mem(self, state, action, reward, new_state, terminal):
        self.memory.store(state, action, reward, new_state, terminal)

    def update_target_nets(self, tau=None):

        if tau is None:
            tau = self.tau

        weights1 = []
        targets1 = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights1.append(weight * tau + targets1[i] * (1 - tau))
        self.target_actor.set_weights(weights1)

        weights2 = []
        targets2 = self.target_crit1.weights
        for i, weight in enumerate(self.crit1.weights):
            weights2.append(weight * tau + targets2[i] * (1 - tau))
        self.target_crit1.set_weights(weights2)

        weights3 = []
        targets3 = self.target_crit2.weights
        for i, weight in enumerate(self.crit2.weights):
            weights3.append(weight * tau + targets3[i] * (1 - tau))
        self.target_crit2.set_weights(weights3)

    def train(self):
        if self.memory.mem_ptr < self.batch_size:
            return
        states, actions, rewards, new_states, terminals = self.memory.sample(self.batch_size)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            target_actions = self.target_actor(new_states)
            # print(target_actions.shape())
            target_actions += tf.clip_by_value(
                tf.random.normal(shape=[*np.shape(target_actions)], mean=0.0, stddev=0.2), 0.5, 0.5)


            target_actions = self.max_action * (tf.clip_by_value(target_actions, self.min_action, self.max_action))

            target_next_state_val = tf.squeeze(self.target_crit1(new_states, target_actions), 1)
            target_next_state_val2 = tf.squeeze(self.target_crit2(new_states, target_actions), 1)

            crit_val = tf.squeeze(self.crit1(states, actions), 1)
            crit_val2 = tf.squeeze(self.crit2(states, actions), 1)

            new_state_target = tf.math.minimum(target_next_state_val, target_next_state_val2)

            target_values = rewards + self.gamma * new_state_target * terminals
            crit1_loss = tf.keras.losses.MSE(target_values, crit_val)
            crit2_loss = tf.keras.losses.MSE(target_values, crit_val2)

        grad1 = tape1.gradient(crit1_loss, self.crit1.trainable_variables)
        grad2 = tape2.gradient(crit2_loss, self.crit2.trainable_variables)

        self.c_opt1.apply_gradients(zip(grad1, self.crit1.trainable_variables))
        self.c_opt2.apply_gradients(zip(grad2, self.crit2.trainable_variables))


        self.train_step +=1

        if self.train_step % self.actor_update_steps == 0:

            with tf.GradientTape() as tape3:
                new_pol_action = self.actor(states)
                actor_loss = -self.crit1(states,new_pol_action)
                actor_loss = tf.math.reduce_mean(actor_loss)

            grad3 = tape3.gradient(actor_loss, self.actor.trainable_variables)
            self.a_opt.apply_gradients(zip(grad3,self.actor.trainable_variables))

            self.update_target_nets()



def main():

    tf.random.set_seed(16)
    agent = TD3Agent(1)
    episodes = 20000
    episode_reward = []
    total_av =[]
    target = False

    for s in range(episodes):
        if target:
            break
        total_reward = 0
        state = env.reset()
        done = False

        while not done:
            # env.render()
            action = agent.choose_act(state)
            next_state, reward, done,_ = env.step(action)
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                episode_reward.append(total_reward)
                avg_reward = np.mean(episode_reward[-100:])
                total_av.append(avg_reward)
                print("total reward after {} steps is {} and avg reward is {}".format(s, total_reward, avg_reward))
                if int(avg_reward) == 90:
                    target = True


    ep = [i for i in range(len(total_av))]
    plt.plot(ep, avg_reward, 'b')
    plt.title("avg rewards vs episodes")
    plt.xlabel("episodes")
    plt.ylabel("avg rewards")
    plt.grid(True)
    plt.show()

    # env = gym.make('MountainCarContinuous-v0')
    # print(len(env.action_space.high))
    # print(env.observation_space.low)
    # print(env.observation_space.high)
    # print(env.action_space.low)
    # print(env.action_space.high)
    # agent = Agent(2)
    # n_games = 1000
    #
    # best_score = env.reward_range[0]
    # score_history = []
    #
    # for i in range(n_games):
    #     observation = env.reset()
    #     terminal = False
    #     score = 0
    #     while not terminal:
    #         action = agent.choose_action(observation)
    #         observation_, reward, done, info = env.step(action)
    #         agent.train()
    #         score += reward
    #         observation = observation_
    #
    #     score_history.append(score)
    #     avg_score = np.mean(score_history[-100:])
    #
    #     if avg_score > best_score:
    #         best_score = avg_score
    #         agent.save_models()
    #
    #     print('episode', i, 'score %1.f' % score, 'average_score % 1.f' % avg_score)


if __name__ == "__main__":
    main()

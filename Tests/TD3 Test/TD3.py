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
#update
import os
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw

import matplotlib.pyplot as plt

from collections import deque
import tensorflow as tf

seed = 16
env = gym.make('MountainCarContinuous-v0')


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


# class CriticNet(keras.Model):
#     def __init__(self, fc1_dims , fc2_dims, name, chckpt_dir = 'tmp/td3'):
#         super(CriticNet,self).__init__()
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.model_name = name
#         self.checkpoint_dir = chckpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name = '_td3')
#
#         self.fc1 = Dense(self.fc1_dims, activation='relu')
#         self.fc2 = Dense(self.fc2_dims, activation = 'relu')
#         self.q = Dense(1, activation= None)
#
#     def call(self, state, action):
#         q1_act_val = self.fc1(tf.concat([state, action], axis=1))
#         q1_act_val = self.fc2((q1_act_val))
#         q = self.q(q1_act_val)
#
#         return q

# class ActorNet(keras.Model):
#     def __init__(self, fc1_dims, fc2_dims,n_actions, name, chckpt_dir='tmp/td3'):
#         super(ActorNet, self).__init__()
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.n_actions = n_actions
#         self.model_name = name
#         self.checkpoint_dir = chckpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name='_td3')
#
#         self.fc1 = Dense(self.fc1_dims,activation='relu')
#         self.fc2 = Dense(self.fc2_dims,activation='relu')
#         self.mu = Dense(self.n_actions, activation='tanh')
#
#     def call(self,state):
#         prob = self.fc1(state)
#         prob = self.fc2(prob)
#         mu = self.mu(prob)
#         return mu


# class Agent:
#     def __init__(self, alpha, beta, input_dims,env,  tau=0.005, gamma=0.99, update_actor_interval=2,
#                  warmup=1000,n_actions=2,max_size=1e5,layer1_size = 256,layer2_size=256,batch_size=64):
#         self.gamma = gamma
#         self.tau = tau
#         self.max_action = env.action_space.high[0]
#         self.min_action = env.action_space.low[0]
#         self.memory = ReplayBuffer(max_size, input_dims, n_actions)
#         self.batch_size = batch_size
#         self.learn_step_cntr = 0
#         self.time_step = 0
#         self.warmup = warmup
#         self.n_actions = n_actions
#         self.update_actor_it = update_actor_interval
#         self.noise = OrnsteinUhlenbeckActionNoise(self.env.action_space.shape[0], seed, theta=0.2,
#                            sigma=0.5)
#
#         self.actor = ActorNet(layer1_size,layer2_size,n_actions=n_actions,
#                               name='actor')
#         self.critic1 = CriticNet(layer1_size,layer2_size,name='critic1')
#         self.critic2 = CriticNet(layer1_size, layer2_size, name='critic2')
#
#         self.target_actor = ActorNet(layer1_size, layer2_size, n_actions=n_actions,
#                               name='target_actor')
#         self.target_critic1 = CriticNet(layer1_size, layer2_size, name='target_critic1')
#         self.target_critic2 = CriticNet(layer1_size, layer2_size, name='target_critic2')
#
#         self.actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
#         self.critic1.compile(optimizer=Adam(learning_rate=beta),loss='mean_squared_error')
#         self.critic2.compile(optimizer=Adam(learning_rate=beta),loss='mean_squared_error')
#
#         self.target_actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
#         self.target_critic1.compile(optimizer=Adam(learning_rate=beta),loss='mean_squared_error')
#         self.target_critic2.compile(optimizer=Adam(learning_rate=beta),loss='mean_squared_error')
#
#         self.update_network_params(tau=1) #hardupdate , first step update = starting of target = start of online net
#
#     def choose_action(self,observation):
#         if self.time_step < self.warmup:
#             mu = np.random.normal(scale = 0.1, size=(self.n_actions,))
#
#         else:
#             state = tf.convert_to_tensor([observation],dtype=tf.float32)
#             mu = self.actor(state)[0]
#         mu_prime = mu + self.noise.sample()
#         mu_prime = tf.clip_by_value(mu_prime,self.min_action, self.max_action)
#
#         self.time_step += 1
#
#         return mu_prime
#
#
#     def remember(self,state,action,reward,new_state,done):
#         self.memory.mem_store(state,action,reward,new_state,done)
#
#     def learn(self):
#         if self.memory.mem_cntr < self.batch_size:
#             return
#
#         states, actions, rewards, new_states, dones = self.memory.sample_mem(self.batch_size)
#
#         states = tf.convert_to_tensor(states, dtype= tf.float32)
#         actions = tf.convert_to_tensor(actions, dtype=tf.float32)
#         rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
#         new_states = tf.convert_to_tensor(new_states, dtype = tf.float32)
#
#         with tf.GradientTape(persistent=True) as tape: #2different applied gradients for single tape
#             target_actions = self.actor(new_states)
#             target_actions = target_actions + tf.clip_by_value(self.noise(), -0.5, 0.5)
#             target_actions = tf.clip_by_value(target_actions, self.min_action ,self.max_action)
#
#             q1_ = self.target_critic1(new_states,target_actions)
#             q2_  = self.target_critic2(new_states,target_actions)
#
#             #shape is [batch_size,1] need [batch_size]
#
#             q1_ = tf.squeeze_(q1_,1)
#             q2_ = tf.squeeze_(q2_,1)
#
#             q1 = tf.squeeze(self.critic1(states,actions),1) #values of states and actions based on crtit1 and crit2
#             q2 = tf.squeeze(self.critic2(states,actions),1)
#
#             critic_value_ = tf.math(q1_,q2_)
#
#             target = rewards + self.gamma*critic_value_*(1-dones)
#
#             critic_1_loss = keras.losses.MSE(target, q1)
#             critic_2_loss = keras.losses.MSE(target,q2)
#
#             critic_1_grad = tape.gradient(critic_1_loss,self.critic1.trainable_variables)
#             critic_2_grad = tape.gradient(critic_2_loss, self.critic2.trainable_variables)
#
#             self.learn_step_cntr +=1
#
#             if self.learn_step_cntr % self.update_actor_it !=0:
#                 return
#
#             with tf.GradientTape() as tape:
#                 new_actions = self.actor(states)
#                 critic_1_value = self.critic_1(states, new_actions)
#                 actor_loss = -tf.math.reduce_mean(critic_1_value)
#
#
#             actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
#             self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))
#
#             self.update_network_params()
#
#     def update_network_params(self, tau = None):
#         if tau == None :
#             tau = self.tau
#
#
#         weights = []
#         targets = self.target_actor.weights
#
#         for i ,weight in enumerate(self.actor.weights):
#             weights.append(weight*tau + targets[i] * (1-tau))
#
#         self.target_actor.set_weights(weights)
#
#         weights = []
#         targets = self.target_critic1.weights
#
#         for i, weight in enumerate(self.critic1.weights):
#             weights.append(weight * tau + targets[i] * (1 - tau))
#
#         self.target_critic1.set_weights(weights)
#
#         weights = []
#         targets = self.target_critic2.weights
#
#         for i, weight in enumerate(self.critic2.weights):
#             weights.append(weight * tau + targets[i] * (1 - tau))
#
#         self.target_critic2.set_weights(weights)
#
#
#     def save_models(self):
#         print('....saving models')
#         self.actor.save_weights(self.actor.checkpoint_file)
#         self.critic1.save_weights(self.critic1.checkpoint_file)
#         self.critic2.save_weights(self.critic2.checkpoint_file)
#         self.target_actor.save_weights(self.target_actor.checkpoint_file)
#         self.target_critic1.save_weights(self.target_critic1.checkpoint_file)
#         self.target_critic2.save_weights(self.target_critic2.checkpoint_file)
#
#
#     def load_models(self):
#         print('.....loading models')
#         self.actor.load_weights(self.actor.checkpoint_file)
#         self.critic1.load_weights(self.critic1.checkpoint_file)
#         self.critic2.load_weights(self.critic2.checkpoint_file)
#         self.target_actor.load_weights(self.target_actor.checkpoint_file)
#         self.target_critic1.load_weights(self.target_critic1.checkpoint_file)
#         self.target_critic2.load_weights(self.target_critic2.checkpoint_file)
#


def main():
    env = gym.make('MountainCarContinuous-v0')
    print(env.observation_space.low)
    print(env.observation_space.high)
    print(env.action_space.low)
    print(env.action_space.high)
    agent = Agent(2)
    n_games = 1000

    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        observation = env.reset()
        terminal = False
        score = 0
        while not terminal:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.train()
            score += reward
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %1.f' % score, 'average_score % 1.f' % avg_score)


if __name__ == "__main__":
    main()

# live_plot = False
#
# seed = 16  # random.randint(0,100) #58 is nice #2021 #78
# num_episodes = 2500
# max_steps = 1000  # Maximum number of steps in an episode
# min_steps = max_steps
# exploring_starts = 1
# average_of = 100
#
# step_decay = 1  # 0.995
# augment = 0  # 0.001
#
# render_list = [0, 150, 300, 800, 1000, 1500, 1800, 2000, 2500]  # 0, 10, 25, 50, 100, 120, 150, 200] #, 300, 500, 1000, 1500, 1997, 1998, 1999, 2000]#0, 10, 20,
# # 30, 40, 50, 100, 110, 120, 130, 140, 150 ]  # 50, 51, 52, 53, 100, 101, 102, 103, 104, 105] #0, 10, 20, 30, 31, 32,
# # 33, 34, 35]
# save = True
#
#
# class Agent:
#
#     epsilon = 0
#     epsilon_min = 0
#     decay = 0.9
#     Learn_start = 1000
#     gamma = 0.99
#     alpha = 0.002
#     tau = 0.005
#
#     mem_len = 1e5
#     memory = deque(maxlen=int(mem_len))
#
#     def __init__(self,env,seed):
#
#         self.env = env
#         random.seed(seed)
#         np.random.seed(seed)
#         tf.random.set_seed(seed)
#         self.noise = 0.1
#         self.env.seed(seed)
#         self.actor = self.createModel()
#         self.critic1 = self.createModel((self.env.observation_space.shape[0], self.env.action_space.shape[0]))
#         self.critic2 = self.createModel((self.env.observation_space.shape[0], self.env.action_space.shape[0]))
#
#         self.target_actor = self.createModel()
#         self.target_critic1 = self.createModel((self.env.observation_space.shape[0], self.env.action_space.shape[0]))
#         self.target_critic2 = self.createModel((self.env.observation_space.shape[0], self.env.action_space.shape[0]))
#         self.reset()
#
#
#
#     def createModel(self, input=None):
#         """Generate neural network models based on inputs, defaults to Actor model"""
#         last_init = tf.random_uniform_initializer(minval=-0.003,
#                                                   maxval=0.003)  # To prevent actor network from causing steep gradients
#         if input is None:
#             input = self.env.observation_space.shape[0]  # Actor
#             inputs = keras.layers.Input(shape=(input,))
#             hidden = keras.layers.Dense(256, activation="relu")(inputs)
#             hidden = keras.layers.Dense(256, activation="relu")(hidden)
#             outputs = keras.layers.Dense(self.env.action_space.shape[0], activation="tanh", kernel_initializer=last_init)(hidden)
#             model = Actor(inputs, outputs)
#             lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#                 initial_learning_rate=self.alpha / 2,
#                 decay_steps=1e9,
#                 decay_rate=1)  # This could allow us to use decaying learning rate
#             model.compile(loss="mean", optimizer=Adam(
#                 learning_rate=lr_schedule))  # Compile model with optimizer so we can apply tape.gradient later
#         else:  # Critic
#             input_o, input_a = input
#             input1 = keras.layers.Input(shape=(input_o,))
#             input2 = keras.layers.Input(shape=(input_a,))
#             input11 = keras.layers.Dense(16, activation="relu")(input1)
#             input11 = keras.layers.Dense(32, activation="relu")(input11)
#             input21 = keras.layers.Dense(32, activation="relu")(input2)
#             cat = keras.layers.Concatenate()([input11, input21])
#             hidden = keras.layers.Dense(256, activation="relu")(cat)
#             hidden = keras.layers.Dense(256, activation="relu")(hidden)
#             outputs = keras.layers.Dense(1, activation="linear", kernel_initializer=last_init)(hidden)
#             lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#                 initial_learning_rate=self.alpha / 1,
#                 decay_steps=1e9,
#                 decay_rate=1)
#             model = Critic([input1, input2], outputs)
#             model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=lr_schedule))  # mean_squared_error
#         return model
#
#     def replayBuffer(self, state, action, reward, next_state, terminal):
#         self.memory.append([state,action,reward,next_state,terminal])

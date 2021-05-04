import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt

env = gym.make("MountainCarContinuous-v0")

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

l_bound = env.action_space.low[0]
u_bound = env.action_space.high[0]


class OUActionnoise:  # noise to allow for random actions
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_dev * np.sqrt(self.dt)
             * np.random.normal(size=self.mean.shape))
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Mem_Buffer:
    def __init__(self, max_mem_size=1e5, batch_size=64):
        self.mem_size = max_mem_size
        self.batch_size = batch_size

        self.mem_cntr = 0

        self.state_mem = np.zeros((self.mem_size, num_states))
        self.action_mem = np.zeros((self.mem_size, num_actions))
        self.reward_mem = np.zeros((self.mem_size, 1))
        self.next_state_mem = np.zeros((self.mem_size, self.num_states))
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.bool)

    # s a r s't
    def store_mem(self, state,action,reward,next_state,terminal):
        idx = self.mem_cntr % self.mem_size

        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.terminal_mem = terminal
        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr,self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_mem[batch]
        next_states = self.next_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        terminals = self.terminal_mem[batch]

        return states, actions, rewards, next_states, terminals

    # def store_mem(self):
    #     self


class Critic_net(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_act=1,
                 name='critic', chkpt_dir='tmp/ddpg'):
        super(Critic_net, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.number_of_actions = n_act
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.check_point_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):  # feedforward
        act_val = self.fc1(tf.concat([state, action]), axis=1)
        act_val = self.fc2(act_val)

        q = self.q(act_val)

        return q


class Actor_net(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, num_act=1,
                 name='Actor', chkpt_dir='tmp/ddpg'):
        super(Actor_net, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_act = num_act

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_act, activation='tanh')

    def call(self, state):
        q2 = self.fc1(state)
        q2 = self.fc2(q2)

        mu = self.mu(q2)

        return mu


class Agent:
    def __init__(self, input_dims, lr_a=0.001, lr_c=0.002, env=None, gamma=0.99, n_actions=1, max_size=1e6,
                 tau=0.005, fc1=400, fc2=300, batch_size=64, noise=0.2):
        self.gamma = gamma
        self.tau = tau
        self.memory = Mem_Buffer(max_size, batch_size)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_a = u_bound
        self.min_a = l_bound

        self.actor = Actor_net(num_act=n_actions, name='actor')
        self.critic = Critic_net(name='critic')
        self.target_actor = Actor_net(num_act=n_actions, name='target_act')
        self.target_critic = Critic_net(name='critic_target')

        self.actor.compile(optimizer=Adam(learning_rate=lr_a))
        self.critic.compile(optimizer=Adam(learning_rate=lr_c))
        self.target_actor.compile(optimizer=Adam(learning_rate=lr_a))
        self.target_critic.compile(optimizer=Adam(learning_rate=lr_c))

        self.update_net_params(tau=1)

    def update_net_params(self, tau=None):
        if tau is None:
            tau = self.tau

        weights_a = []
        targets_a = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights_a.append(weight * tau + targets_a[i] * (1 - tau))
        self.target_actor.set_weights(weights_a)

        weights_c = []
        targets_c = self.target_critic.weights_c
        for i, weight in enumerate(self.critic.weights):
            weights_c.append(weight * tau + targets_c[i] * (1 - tau))
        self.target_critic.set_weights(weights_c)

    def remember(self, state,action,reward,new_state,terminal):
        self.memory.store_mem(state,action,reward,new_state,terminal)

    def save_models(self):
        print('>>>>>>>saving>>>>>>>>')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.check_point_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.check_point_file)

    def load_models(self):
        print('>>>>>loading>>>>>>>>>')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.check_point_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.check_point_file)

    def choose_act(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.n_actions)
        actions = tf.clip_by_value(actions, self.min_a, self.max_a)
        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, terminal = self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions),1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = reward + self.gamma * critic_value_ * (1 - done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_grad, self.actor.trainable_variables))


if __name__ == '__main__':
    agent = Agent(input_dims=env.observation_space.shape, env = env, n_actions= env.action_space.shape[0])
    num_episodes = 500

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_,reward,terminal,_ = env.step(action)
            agent.remember(observation,action,reward,observation_,terminal)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for i in range(num_episodes):
        observation = env.reset()
        terminal = False
        score = 0
        while not terminal:
            action = agent.choose_act(observation, evaluate)
            observation_,reward, done,_ = env.step(action)
            score += reward
            agent.remember(observation,action,reward,observation_,terminal)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                if not load_checkpoint:
                    agent.save_models()
            print('episode ', i , 'score %.1f' %score, 'avg score % 1f' %avg_score)




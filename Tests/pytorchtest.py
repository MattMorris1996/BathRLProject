import math
import random
import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

env = gym.make("MountainCarContinuous-v0")

np.random.seed(9)
env.seed(9)
action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]
# class ActorCritic(nn.Module):
#     def __init__(self, num_inputs, num_outputs, hidden_size, std = 0.0):
#         super(ActorCritic, self).__init__()
#
#         self.critic = nn.Sequential(
#             nn.Linear(num_inputs, hidden_size),
#             nn.ReLU(),
#             nn.Linear()
#
#
#         )

class Actor(nn.Module):
    def __init__(self,state_size,action_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_size, 32)
        self.linear2 = nn.Linear(32,32)
        self.linear3 = nn.Linear(32, action_size)
        self.reset_params()

    def reset_params(self):
        self.linear1.weight.data.normal_(0, 1e-1)
        self.linear2.weight.data.normal_(0, 1e-1)
        self.linear3.weight.data.normal_(0, 1e-2)

    def forward(self,state):
        x = state
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return torch.tanh(x)

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_size + action_size, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear1.weight.data.normal_(0, 1e-1)
        self.linear2.weight.data.normal_(0, 1e-1)
        self.linear3.weight.data.normal_(0, 1e-2)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

class Memory:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        return list(zip(*random.sample(self.memory, self.batch_size)))

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, state_size, action_size, buffer_size, batch_size, gamma, tau):
        # Actor Network and Target Network
        self.actor = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        # Critic Network and Target Network
        self.critic = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # copy weights
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.memory = Memory(buffer_size, batch_size)
        self.gamma = gamma
        self.tau = tau
        self.sd = 1

    def hard_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(self, batch):

        state, action, reward, next_state, done = batch

        state = torch.tensor(state).to(device).float()
        next_state = torch.tensor(next_state).to(device).float()
        reward = torch.tensor(reward).to(device).float()
        action = torch.tensor(action).to(device)
        done = torch.tensor(done).to(device).int()

        # update critic
        next_action = self.actor_target(next_state)

        Q_target = self.critic_target(next_state, next_action).detach()
        Q_target = reward.unsqueeze(1) + (self.gamma * Q_target * ((1 - done).unsqueeze(1)))

        critic_loss = F.mse_loss(self.critic(state, action), Q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor

        action_prediction = self.actor(state)
        actor_loss = -self.critic(state, action_prediction).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update actor_target and critic_target

        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)

    def act(self, state, noise=True):
        state = torch.tensor(state).to(device).float()
        action = self.actor(state).cpu().data.numpy()

        if noise:
            noise = np.random.normal(0, self.sd)
            action = action + noise

        if action[0] > 1:
            action[0] = 1
        if action[0] < -1:
            action[0] = -1
        return action

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        if len(self.memory) >= self.memory.batch_size:
            self.learn(self.memory.sample())

    def save(self):
        torch.save(self.actor, "actor.pkl")
        torch.save(self.critic, "critic.pkl")

    def test(self):
        new_env = make("MountainCarContinuous-v0")
        new_env.seed(9)
        reward = []
        for i in range(50):
            state = new_env.reset()
            local_reward = 0
            done = False
            while not done:
                action = self.act(state, noise=False)
                state, r, done, _ = new_env.step(action)
                local_reward += r
            reward.append(local_reward)
        return reward

def DDPG(episodes = 500):
    agent = Agent(state_size= state_size,action_size= action_size,buffer_size=int(1e6), batch_size= 64, gamma= 0.99,
                   tau= 1e-3)
    reward_list = []
    mean_reward = -20000
    for i in range(episodes):
        state = env.reset()
        total_reward = 0
        terminal = False
        while not terminal:
            env.render()
            action = agent.act(state)
            next_state, reward, terminal, _ = env.step(action)
            agent.step(state,action,reward,next_state,terminal)
            total_reward += reward
            state = next_state

            reward_list.append(total_reward)
            agent.sd = max(agent.sd - 0.01, 0.1)
            if total_reward > 50:
                r = agent.test()
                local_mean = np.mean(r)
                print(f"episodeL {i + 1}, current reward: {total_reward} , max reward: {np.max(r)},"
                      f" mean reward: {local_mean}")

                if local_mean > mean_reward:
                    mean_reward = local_mean
                    agent.save()
                    print("Saved")

            else:
                print(f"episode: {i+1}, current reward: {total_reward}")
    return reward_list








if __name__ == "__main__":
    reward = DDPG()
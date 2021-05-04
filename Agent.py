from collections import deque

import numpy as np
import gym
import math

from DDPG import DDPG
from TD3 import TD3
from AgentLogger import AgentLogger


class Agent:
    exploring_starts = 1
    reward_average_pool_size = 100

    environment = 'MountainCarContinuous-v0'

    def __init__(self, agent_num, num_episodes, seed, logger: AgentLogger = None, max_steps=1000):
        self.num_episodes = num_episodes
        self.seed = seed
        self.max_steps = max_steps
        self.agent_num = agent_num
        self.logger = logger

    def train(self):
        env = gym.make(self.environment).env
        agent = TD3(env, self.seed)

        rewards = np.zeros(self.num_episodes)
        reward_pool = deque(maxlen=self.reward_average_pool_size)
        averages = np.ones(self.num_episodes)

        for episode in range(self.num_episodes):
            state = env.reset()

            total_reward = 0
            step = 0
            for step in range(self.max_steps):

                if self.logger:
                    self.logger.save_frames(env.render, episode)

                action = agent.choose_action(state, scale=episode < self.exploring_starts)

                next_state, reward, terminal, info = env.step(action)
                next_state = next_state  # ?
                agent.train(state, action, reward, next_state, terminal, step)

                total_reward += reward
                state = next_state

                if terminal:
                    break

            env.close()
            agent.noise.reset()

            rewards[episode] = total_reward
            reward_pool.append(total_reward)
            moving_average_reward = np.mean(reward_pool)
            averages[episode] = moving_average_reward

            solved = moving_average_reward >= 90
            if solved:
                pass
                # break

            if self.logger:
                self.logger.log(
                    self.agent_num,
                    episode,
                    step,
                    total_reward,
                    moving_average_reward,
                    solved,
                )

        if self.logger:
            self.logger.pickle_dump(self.agent_num)

        return rewards

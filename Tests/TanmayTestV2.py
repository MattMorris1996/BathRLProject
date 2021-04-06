import gym

import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque

seed = 1
num_episodes = 1000
max_steps = 500


class Agent:

    epsilon = 1
    gamma = 0.85
    alpha = 0.01
    tau = 1  # 0.15
    decay = 0.995
    noise = 0
    memory = deque(maxlen=int(1e5))

    def __init__(self, env, seed):

        self.env = env
        random.seed(seed)

        self.model = self.createModel()
        self.target_model = self.createModel()

        self.reset()

    def createModel(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(1))  # self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.alpha))
        return model

    def replayBuffer(self, state, action, reward, new_state, terminal):
        self.memory.append([state, action, reward, new_state, terminal])

    def replay(self):
        batch_size = 32
        state_update = np.zeros([batch_size, 2])
        target_update = np.zeros(batch_size)
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for idx, sample in enumerate(samples):
            state, action, reward, new_state, terminal = sample
            target = self.target_model.predict(state)
            # print(target)
            if terminal:
                # target[0][action] = reward
                target[0] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                # target[0][action] = reward + Q_future * self.gamma
                target[0] = reward + Q_future * self.gamma
            state_update[idx] = state
            target_update[idx] = target[0]  # [action]
        self.model.fit(state_update, target_update, epochs=1, verbose=0)

    def trainTarget(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def train(self, state, action, reward, next_state, terminal, steps):
        self.replayBuffer(state, action, reward, next_state, terminal)
        if steps % 25 == 0:
            self.replay()
        if steps % 50 == 0:
            self.trainTarget()

    def reset(self):
        pass

    def chooseAction(self, state):
        self.epsilon *= self.decay
        self.epsilon = max(self.epsilon/1000, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # else:
            # action_us = np.argmax(self.model.predict(state)[0])
            # action = action_us*2 - 1
        return np.argmax(self.model.predict(state)[0])  # action


def scale(action):
    return [(action*2)-1]


def main():
    env = gym.make('MountainCarContinuous-v0')
    agent = Agent(env, seed)
    rewards = np.zeros(num_episodes)
    for episode in range(num_episodes):
        action = np.zeros(1)
        state = env.reset().reshape(1, 2)
        total_reward = 0
        for step in range(max_steps):
            # env.render()
            if step % 10 == 0:
                # env.render()
                pass
            action[0] = agent.chooseAction(state)
            # print(state, action)
            # print(step)
            # print(total_reward)
            next_state, reward, terminal, info = env.step(action)  # env.action_space.sample())  # take a random action
            next_state = next_state.reshape(1, 2)
            agent.train(state, action, reward, next_state, terminal, step)
            state = next_state
            total_reward += reward

            if terminal:
                break
        # env.close()
        # print(total_reward)
        rewards[episode] = total_reward
        if step >= 199:
            print("Failed to complete in episode {}".format(episode))
            if step % 10 == 0:
                # dqn_agent.save_model("trial-{}.model".format(trial))
                pass
        else:
            print("Completed in {} episodes, with reward of {}, average reward of {}".format(episode, total_reward, rewards.mean()))
            # dqn_agent.save_model("success.model")
            # terminal = False
            # state = env.reset().reshape(1, 2)
            # while not terminal:
            #     env.render()
            #     action = agent.chooseAction(state)
            #     next_state, reward, terminal, info = env.step(action)  # env.action_space.sample())  # take a random action
            #     next_state = next_state.reshape(1, 2)
            #     state = next_state

            # break

    # env.close()


if __name__ == "__main__":
    main()

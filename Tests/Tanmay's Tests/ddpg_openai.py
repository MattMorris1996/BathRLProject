import gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# import pickle

env = gym.make("MountainCarContinuous-v0")

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=400000, log_interval=10)
model.save("ddpg_car")
env = model.get_env()

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_car")
# rewards = np.zeros(num_episodes)
# rewards_av = deque(maxlen=int(average_of))
# averages = np.ones(num_episodes)  # * (-max_steps)
obs = env.reset()
# total_reward = 0
while True:
    action, _states = model.predict(obs)
    # env.render()
    obs, rewards, dones, info = env.step(action)
    # total_reward += rewards
    # if dones:
    #     print(total_reward)
    #     total_reward = 0
    # env.render()
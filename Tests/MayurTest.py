import gym
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color, transform, io

env = gym.make('CarRacing-v0')
print(env.observation_space)
print(env.observation_space.shape)
print(env.action_space)
print(env.action_space.shape)


class imagepreprocess:
    def processing(obs):
        # plt.imshow(obs)
        # plt.show()

        obs1 = obs.astype(np.uint8)
        grey_obs = color.rgb2grey(obs1)
        grey_obs[84:95, 0:12] = 0
        grey_obs[abs(grey_obs - 0.68616) < 0.0001] = 1
        grey_obs[abs(grey_obs - 0.75630) < 0.0001] = 1

        # plt.imshow(grey_obs, cmap='gray')
        # plt.imshow()

        return 2 * grey_obs - 1

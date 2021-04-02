import time

import gym
# import time
import numpy as np
import matplotlib.pyplot as plt

print("State modification testing")


def grey(array):
    # array.transpose(2, 0, 1).reshape(3, -1)
    greyscale = np.dot(array, [0.2989, 0.5870, 0.1140])  # [1/3, 1/3, 1/3])
    return greyscale


def crop(array, y_max=96, y_min=0, x_min=0, x_max=96):
    array_cropped = array[y_min:y_max, x_min:x_max]
    return array_cropped


def main():
    env = gym.make('CarRacing-v0')
    env.reset()
    plt.ion()
    for _ in range(1000):
        env.render()
        state, reward, terminal, a = env.step(env.action_space.sample())  # take a random action
        greyscale = grey(state)
        greyscale_c = crop(greyscale, 84, 0, 24, 72)
        states = crop(greyscale, 96, 84, 12, 96)
        # print(a)
        plt.figure(1)
        plt.imshow(greyscale_c, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        # t = plt.show()
        plt.figure(2)
        plt.imshow(states, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        plt.show()
        time.sleep(1/60)
        plt.close('all')
    env.close()
    pass


if __name__ == "__main__":
    main()

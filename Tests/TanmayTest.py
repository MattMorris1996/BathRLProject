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
    return array[y_min:y_max, x_min:x_max]
    # return array_cropped

def compute_steering_speed_gyro_abs(a):
    right_steering = a[4, 36:46].mean() / 255
    left_steering = a[4, 26:36].mean() / 255
    steering = (right_steering - left_steering + 1.0) / 2

    left_gyro = a[4, 46:60].mean() / 255
    right_gyro = a[4, 60:76].mean() / 255
    gyro = (right_gyro - left_gyro + 1.0) / 2

    speed = a[:, 1][:-2].mean() / 255
    abs1 = a[:, 6][:-2].mean() / 255
    abs2 = a[:, 8][:-2].mean() / 255
    abs3 = a[:, 10][:-2].mean() / 255
    abs4 = a[:, 12][:-2].mean() / 255

    # white = np.ones((round(speed * 100), 10))
    # black = np.zeros((round(100 - speed * 100), 10))
    # speed_display = np.concatenate((black, white))*255

    # cv2.imshow('sensors', speed_display)
    # cv2.waitKey(1)

    return [steering, speed, gyro, abs1, abs2, abs3, abs4]

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
        steering, speed, gyro, abs1, abs2, abs3, abs4 = compute_steering_speed_gyro_abs(states)
        print(steering, speed)#, gyro, abs1, abs2, abs3, abs4)
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
    #main()
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])


    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation


    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0


    env = gym.make('CarRacing-v0')
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            greyscale = grey(s)
            greyscale_c = crop(greyscale, 84, 0, 24, 72)
            states = crop(greyscale, 96, 84, 12, 96)
            steering, speed, gyro, abs1, abs2, abs3, abs4 = compute_steering_speed_gyro_abs(states)
            print(steering, speed, gyro, abs1, abs2, abs3, abs4)
            # print(a)
            plt.figure(1)
            plt.imshow(greyscale_c, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
            # t = plt.show()
            plt.figure(2)
            plt.imshow(states, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
            #plt.show()
            #plt.close('all')
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
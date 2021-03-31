import gym


def main():
    env = gym.make('CarRacing-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()
    pass


if __name__ == "__main__":
    main()

import pickle
import matplotlib.pyplot as plt

class DataStore:
    def __init__(self, averages, rewards):
        self.averages = averages
        self.rewards = rewards

def main():
    try:
        with open('data_0.pk1', 'rb') as qt:
            data = pickle.load(qt)
    except:
        # data = DataStore()
        pass
    try:
        with open('data_2.pk1', 'rb') as qt:
            data2 = pickle.load(qt)
    except:
        # data = DataStore()
        pass
    plt.subplot(2, 2, 1)
    plt.plot(data.averages[:1100])
    plt.subplot(2, 2, 2)
    plt.plot(data.rewards[:1100])
    plt.subplot(2, 2, 3)
    plt.plot(data2.averages[:850])
    plt.subplot(2, 2, 4)
    plt.plot(data2.rewards[:850])
    plt.show()

if __name__ == "__main__":
    # print(tf.__version__)
    # print(device_lib.list_local_devices())
    main()


# data_0: buffer;1.5e4, theta;0.2, mu;0.25, augment;0.001, max_step;3000
# data_1: bufffer;2.4e4, theta;0.1, mu;0.15, augment;0.01
# data_2: buffer;5e3
# data_3: buffer;15e3, theta;0.08, mu;0.125, augment;0.01
# data_4: max_step;1000
# data_5: max_step;5000, buffer;2.5e4
# data_6: buffer;4.5e4, max_step;1500
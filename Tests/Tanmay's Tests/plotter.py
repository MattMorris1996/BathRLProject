import pickle
import matplotlib.pyplot as plt

class DataStore:
    def __init__(self, averages, rewards):
        self.averages = averages
        self.rewards = rewards

def main():
    try:
        with open('data2.pk1', 'rb') as qt:
            data = pickle.load(qt)
    except:
        # data = DataStore()
        pass
    plt.subplot(2, 1, 1)
    plt.plot(data.averages[:3000])
    plt.subplot(2, 1, 2)
    plt.plot(data.rewards[:3000])
    plt.show()

if __name__ == "__main__":
    # print(tf.__version__)
    # print(device_lib.list_local_devices())
    main()
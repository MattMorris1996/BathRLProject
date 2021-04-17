import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# plot = False
plot = True
# anim_plot = False
anim_plot = True
# plot_opts = [0, 10, 6]
# plot_opts = [6, 13, 10, 17, 18]
plot_opts = ['ddpg']#, 'ddpg_success1']
# plot_opts = list(range(0,15))
tests = 19

# data_0: buffer;1.5e4, theta;0.2, mu;0.25, augment;0.001, max_step;3000, actor_lr;/100, critic_lr;/10, eps;decay(0.999)
# data_1: bufffer;2.4e4, theta;0.1, mu;0.15, augment;0.01
# data_2: buffer;5e3
# data_3: buffer;15e3, theta;0.08, mu;0.125, augment;0.01
# data_4: max_step;1000
# data_5: max_step;5000, buffer;2.5e4
# data_6: buffer;4.5e4, max_step;1500, theta;0.08, mu;0.125, augment;0.01
# data_7: max_step;10000, buffer;5e4
# data_8: max_step;1000, buffer;5e4, theta;0.08, mu;0.125, augment;0.01
# data_9: max_step;decay(10000,500, 0.99), buffer;4.5e4, theta;0.08, mu;0.125, augment;0.01
# data_10: augment;0
# data_11: max_step;decay(10000,500, 0.995), buffer;4.5e4, theta;0.08, mu;0.125, augment;0.01
# data_12: augment;0, theta;0.08, mu;0.125
# data_13: data_6 - Long Run
# data_14: max_step;decay(10000,1500, 0.995), buffer;4.5e4, theta;0.08, mu;0.125, augment;0.01
# data_15: data_13;rerun, actor_lr;/50
# data_16: data_15;rerun, actor_lr;/200, critic_lr;/50
# data_17: data_6;rerun, augment;0.001
# data_18: data_6;rerun, eps;decay(0.995), augment;0.001
# data_19: data_6;rerun, eps;decay(0.995), augment;0

class DataStore:
    def __init__(self, averages, rewards):
        self.averages = averages
        self.rewards = rewards

def main():
    print("")
    print("All Data info:")
    print("")

    for value in range(0,tests+1):
        try:
            with open('data_{}.pk1'.format(value), 'rb') as qt:
                data_temp = pickle.load(qt)
                print("Max reward for Test {:2}: {:8.3f}, Max average reward for Test {:2}: {:8.3f}".format(value, max(data_temp.rewards), value, max(data_temp.averages)))
        except:
            pass

    try:
        with open('data_ddpg.pk1', 'rb') as qt:
            data_temp = pickle.load(qt)
            print("Max reward for DDPG: {:8.3f}, Max average reward for DDPG: {:8.3f}".format(max(data_temp.rewards), max(data_temp.averages)))
    except:
        pass

    if plot:
        if anim_plot:
            for dat in range(0,len(plot_opts)):
                try:
                    with open('data_{}.pk1'.format(plot_opts[dat]), 'rb') as qt:
                        data = pickle.load(qt)
                        point_a = 1000
                        for limit, point in enumerate(data.averages):
                            if point == point_a:
                                break
                            else:
                                point_a = point
                        fig, (ax1, ax2) = plt.subplots(2,1)
                        ax1.set_xlabel("Episode")
                        ax1.set_ylabel("Episode Reward")
                        ax2.set_xlabel("Episode")
                        ax2.set_ylabel("Episode Reward")
                        for ax in [ax1, ax2]:
                            ax.set_ylim(-100, 100)
                            ax.set_xlim(0, limit)
                            # ax.grid()
                        # ax1 = ax1.axes(xlim=(0, limit), ylim=(-100, 100))
                        # ax2 = ax2.axes(xlim=(0, limit), ylim=(-100, 100))
                        line1, = ax1.plot([], [], lw=2)
                        line2, = ax2.plot([], [], lw=2)
                        line = [line1, line2]
                        # ax = [ax1, ax2]
                        def init_graph():
                            line[0].set_data([], [])
                            line[1].set_data([], [])
                            return line

                        # animation function.  This is called sequentially
                        def animate(i):
                            x = np.linspace(0, i, i)
                            y1 = data.averages[:i]
                            line[0].set_data(x, y1)
                            y2 = data.rewards[:i]
                            line[1].set_data(x, y2)
                            return line

                        plt.tight_layout()
                        anim = animation.FuncAnimation(fig, animate, init_func=init_graph,
                                                       frames=limit, interval=20, blit=True)
                        anim.save('./videos/learning_curve_{}.mp4'.format(plot_opts[dat]), fps=30,
                                  extra_args=['-vcodec', 'libx264'])

                        plt.show()

                except:
                    pass

        else:
            for dat in range(0,len(plot_opts)):
                try:
                    with open('data_{}.pk1'.format(plot_opts[dat]), 'rb') as qt:
                        data = pickle.load(qt)
                        point_a = 1000
                        for limit, point in enumerate(data.averages):
                            if point == point_a:
                                break
                            else:
                                point_a = point
                        plt.subplot(len(plot_opts), 2, 2*dat+1)
                        plt.plot(data.averages[:limit])
                        plt.xlabel("Episode")
                        plt.ylabel("Avg. Epsiodic Reward (of 100)")
                        plt.subplot(len(plot_opts), 2, 2*dat+2)
                        plt.plot(data.rewards[:limit])
                        plt.xlabel("Episode")
                        plt.ylabel("Epsiode Reward")
                except:
                    pass

        # try:
        #     with open('data_ddpg.pk1'.format(plot_opts[dat]), 'rb') as qt:
        #         data = pickle.load(qt)
        #         plt.figure()
        #         plt.subplot(2, 1, 1)
        #         plt.plot(data.averages)
        #         plt.subplot(2, 1, 2)
        #         plt.plot(data.rewards)
        # except:
        #     pass
        plt.show()


# initialization function: plot the background of each frame

if __name__ == "__main__":
    main()


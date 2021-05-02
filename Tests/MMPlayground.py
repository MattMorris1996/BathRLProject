import gym
import numpy as np
import tensorflow as tf
import pickle


class TileCoding:
    def __init__(self, n_grids, feature_range1, feature_range2):
        self.grids = []
        self.n_grids = n_grids
        for grid in range(n_grids):
            i_grid = TileGrid((8, 8), feature_range1, feature_range2)
            x_offset = i_grid.grid_width
            y_offset = i_grid.grid_height
            i_grid.set_offset(
                (np.random.uniform(-x_offset / 2, x_offset / 2), np.random.uniform(-y_offset / 2, y_offset / 2)))
            self.grids.append(
                i_grid
            )

    def get_feature_vector(self, val1, val2, action):
        feature_vec = np.zeros(64 * self.n_grids + 3)
        feature_vec[action] = 1
        for nth_grid, grid in enumerate(self.grids):
            index = nth_grid * 64 + grid.get_overlap_index(val1, val2) + 3
            feature_vec[index] = 1
        return feature_vec


class TileGrid:
    def __init__(self, grid_size, variable_range1, variable_range2):
        self.n_columns, self.n_rows = grid_size

        self.v1_min, self.v1_max = variable_range1
        self.v2_min, self.v2_max = variable_range2

        standard_width = (self.v1_min - self.v1_max) / self.n_columns

        self.grid_width = (standard_width * (self.n_columns + 1)) / self.n_columns

        standard_height = (self.v2_min - self.v2_max) / self.n_rows
        self.grid_height = (standard_height * (self.n_rows + 1)) / self.n_rows

        self.scale_increase_v1 = standard_width / 2
        self.scale_increase_v2 = standard_height / 2

        self.v1_grid_min, self.v1_grid_max = (self.v1_min - self.scale_increase_v1,
                                              self.v1_max + self.scale_increase_v1)

        self.v2_grid_min, self.v2_grid_max = (self.v2_min - self.scale_increase_v2,
                                              self.v2_max + self.scale_increase_v2)

        self.grid_offset_x, self.grid_offset_y = (np.random.uniform(-self.scale_increase_v1, self.scale_increase_v1),
                                                  np.random.uniform(-self.scale_increase_v2, self.scale_increase_v2))

    def set_offset(self, value):
        self.grid_offset_x, self.grid_offset_y = value

    def get_overlap_index(self, value1, value2):
        col = (value1 - self.v1_grid_min + self.grid_offset_x) // self.grid_width
        row = (value2 - self.v2_grid_min + self.grid_offset_y) // self.grid_height
        index = int(row + col * self.n_columns)
        return index


GAMMA = 0.95
ALPHA = 0.5/8


class QTable:
    def __init__(self):
        self.grid_size = self.grid_columns, self.grid_rows = 8, 8

        self.n_grids = 8
        self.n_actions = 3

        self.tile_coder = TileCoding(self.n_grids, (-1.2, 0.6), (-0.07, 0.07))
        self.parameter_vector = np.zeros(self.n_grids * self.grid_rows * self.grid_columns + self.n_actions,
                                         dtype=np.float64)

    def _get_value(self, state, action):
        v1, v2 = state
        vect = self.tile_coder.get_feature_vector(v1, v2, action)
        return self.parameter_vector.dot(vect)

    def get_optimal_policy(self, state):
        values = np.zeros((3, 1), dtype=np.float64)
        for action in range(3):
            values[action] = self._get_value(state, action)

        prob = np.random.uniform(0, 1)
        if prob < 0.1:
            return np.random.randint(3)
        else:
            return np.argmax(values)

    def _gradient(self, state, action):
        def linear_function(w, f):
            return tf.tensordot(w, f, axes=1)

        weights = tf.Variable(self.parameter_vector, dtype=tf.float32)

        v1, v2 = state
        features = tf.constant(self.tile_coder.get_feature_vector(v1, v2, action), dtype=tf.float32)

        with tf.GradientTape() as tape:
            y = linear_function(weights, features)

        gradients = tape.gradient(y, weights)

        return gradients.numpy()

    def update(self, state, action, next_state, next_action, reward):
        error = reward + (GAMMA * self._get_value(next_state, next_action) - self._get_value(state, action))
        v1, v2 = state
        gradient = self.tile_coder.get_feature_vector(v1, v2, action)
        self.parameter_vector += ALPHA * error * gradient

    def update_terminal(self, state, action, reward):
        error = reward - self._get_value(state, action)
        v1, v2 = state
        gradient = self.tile_coder.get_feature_vector(v1, v2, action)
        self.parameter_vector += ALPHA * error * gradient


def learn():
    N_EPISODES = 1000
    N_AGENTS = 1
    env = gym.make('MountainCar-v0').env
    agent_returns = np.zeros((N_AGENTS, N_EPISODES))
    try:
        with open('mountain_car_q_table.pk1', 'rb') as qt:
            q = pickle.load(qt)
    except:
        q = QTable()

    for n_agent in range(N_AGENTS):
        for n_episode in range(N_EPISODES):
            print(n_episode)
            observation = env.reset()
            action = q.get_optimal_policy(observation)
            total_return = 0
            steps = 0
            while True:
                steps += 1
                next_observation, reward, done, info = env.step(action)
                total_return += reward
                # env.render()
                if done:
                    q.update_terminal(observation, action, reward)
                    break
                next_action = q.get_optimal_policy(next_observation)
                q.update(observation, action, next_observation, next_action, reward)
                action = next_action
                observation = next_observation
            agent_returns[0][n_episode] = total_return
            print("steps:", str(steps))
        env.close()

    import matplotlib.pyplot as plt

    print(agent_returns)
    plt.plot(np.mean(agent_returns, axis=0))
    plt.show()

    with open('mountain_car_q_table.pk1', 'wb') as handle:
        pickle.dump(q, handle, pickle.HIGHEST_PROTOCOL)

    print(q.parameter_vector)


learn()

import gym
import numpy as np


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

        self.max_rect_width = (self.v1_max - self.v1_min) / (self.n_columns - 1)
        self.max_rect_height = (self.v2_max - self.v2_min) / (self.n_rows - 1)

        self.scale_increase_v1 = self.max_rect_width / 2
        self.scale_increase_v2 = self.max_rect_height / 2

        self.v1_grid_min, self.v1_grid_max = (self.v1_min - self.scale_increase_v1 / 2,
                                              self.v1_max + self.scale_increase_v1 / 2)

        self.v2_grid_min, self.v2_grid_max = (self.v2_min - self.scale_increase_v2 / 2,
                                              self.v2_max + self.scale_increase_v2 / 2)

        self.grid_width, self.grid_height = (self.v1_grid_max - self.v1_grid_min / self.n_columns,
                                             self.v2_grid_max - self.v2_grid_min / self.n_rows)

        self.grid_offset_x, self.grid_offset_y = (0, 0)

    def set_offset(self, value):
        self.grid_offset_x, self.grid_offset_y = value

    def get_overlap_index(self, value1, value2):
        col = (value1 - self.v1_grid_min + self.grid_offset_x) // self.grid_width
        row = (value2 - self.v2_grid_min + self.grid_offset_y) // self.grid_height

        index = int(row + col * self.n_columns)
        return index


class QTable:
    def __init__(self):
        self.grid_size = self.grid_columns, self.grid_rows = 8, 8

        self.n_grids = 10
        self.n_actions = 3

        self.tile_coder = TileCoding(self.n_grids, (-1.2, 0.6), (-.07, .07))
        self.parameter_vector = np.zeros(self.n_grids * self.grid_rows * self.grid_columns + self.n_actions)

    def _get_value(self, state, action):
        v1, v2 = state
        vect = self.tile_coder.get_feature_vector(v1, v2, action)
        return self.parameter_vector.dot(vect)

    def get_optimal_policy(self, state):
        values = np.zeros((3, 1), dtype=np.float64)
        for action in range(self.n_actions):
            values[action] = self._get_value(state, action)

        return np.argmax(values)

    def _gradient(self, state, action):
        h = 0.001
        gradients = np.zeros(self.parameter_vector.size)

        for i, val in enumerate(self.parameter_vector):

            temp = self.parameter_vector[i]

            self.parameter_vector[i] -= h/2
            current_val = self._get_value(state, action)

            self.parameter_vector[i] += h
            new_val = self._get_value(state, action)

            self.parameter_vector[i] = temp
            gradients[i] = (new_val - current_val) / h

        return gradients

    def update(self, state, action, next_state, next_action, reward):
        error = reward + (self._get_value(next_state, next_action) * 0.8 - self._get_value(state, action))
        gradient = self._gradient(state, action)
        self.parameter_vector += error * 0.2 * gradient

    def update_terminal(self, state, action, reward):
        error = reward - self._get_value(state, action)
        gradient = self._gradient(state, action)
        self.parameter_vector += error * gradient


def learn():
    env = gym.make('MountainCar-v0').env
    q = QTable()
    N_EPISODES = 10
    for n_episode in range(N_EPISODES):
        print(n_episode)
        observation = env.reset()
        action = q.get_optimal_policy(observation)
        while True:
            next_observation, reward, done, info = env.step(action)
            if done:
                q.update_terminal(observation, action, reward)
                break
            next_action = q.get_optimal_policy(next_observation)
            q.update(observation, action, next_observation, next_action, reward)
            action = next_action
            observation = next_observation
    env.close()
    print(q.parameter_vector)


learn()

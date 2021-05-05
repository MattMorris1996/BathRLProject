import numpy as np
import matplotlib.pyplot as plt

import os

import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw
import pandas as pd


class AgentLogger:
    def __init__(
            self,
            _saving_interval: int = 25,
            _console_log: bool = False,
            _pickle_log: bool = True,
            _live_plot: bool = False,
            _render_recording: bool = True,
            _render_list=None
    ):
        """Class which separates logging logic from an agent class, an object with this class type is passed as an
        argument to the agent class to handle the logging of data_initial_test to the console or to files.

        Args:
            _saving_interval : int determines the interval in which episodes are logged to a pickle file
            _console_log: bool binary flag which activates the logging of data_initial_test to the console
            _pickle_log: bool binary flag which activates the logging of data_initial_test to a pickle file
            _live_plot: bool binary flag which activates the real time logging of data_initial_test to live matplotlib plot
            _render_recording: bool binary flag which activates the saving of gym frames to a gif file
            _render_list: list contains integers with the index of episodes that should be rendered
        """

        self.frame_buffer = []

        if _render_list is None:
            _render_list = []

        self.saving_interval = _saving_interval
        self.console_log = _console_log
        self.pickle_log = _pickle_log
        self.live_plot = _live_plot
        self.render_recording = _render_recording
        self.render_list = _render_list

        self.columns = [
            'AGENT_ID',
            'NTH_EPISODE',
            'STEPS_TAKEN',
            'TOTAL_REWARD',
            'MOVING_AVERAGE_REWARD',
            'SOLVED',
        ]

        self.data_frame = pd.DataFrame(columns=self.columns)

    def save_frames(self, render_function_cb, nth_episode):
        if nth_episode in self.render_list:
            render_function_cb()
            if self.render_recording:
                frame = render_function_cb(mode='rgb_array')
                im = Image.fromarray(frame)
                drawer = ImageDraw.Draw(im)

                if np.mean(im) < 128:
                    text_color = (255, 255, 255)
                else:
                    text_color = (0, 0, 0)

                drawer.text((im.size[0] / 20, im.size[1] / 18), f'Episode: {nth_episode}', fill=text_color)
                self.frame_buffer.append(im)

    def _live_plot_out(self, agent_id: int, nth_episode: int):
        df = self.data_frame[self.data_frame['AGENT_ID'] == agent_id]

        moving_averages = np.pad(df['MOVING_AVERAGE_REWARD'].to_numpy(), (0, 250 - nth_episode), 'constant')
        rewards = np.pad(df['TOTAL_REWARD'].to_numpy(), (0, (250 - nth_episode)), 'constant')

        plt.subplot(2, 1, 1)
        plt.plot(moving_averages)
        plt.subplot(2, 1, 2)
        plt.plot(rewards)
        plt.pause(0.0001)
        plt.clf()

    def pickle_dump(self, agent_id: int):
        df = self.data_frame[self.data_frame['AGENT_ID'] == agent_id]
        pd.to_pickle(df, './data_initial_test/data_ddpg_agent{}.pk1'.format(agent_id))

    def _episode_log_out(self, agent_id: int):
        df = self.data_frame[self.data_frame['AGENT_ID'] == agent_id]
        row = df.iloc[-1]
        if row['SOLVED']:
            print("Completed in {:4} episodes, "
                  "with reward of {:8.3f}, "
                  "average reward of {:8.3f}"
                  .format(row['NTH_EPISODE'], row['TOTAL_REWARD'], row['MOVING_AVERAGE_REWARD']))
        else:
            print("Failed to complete in episode {:4} "
                  "with reward of {:8.3f} "
                  "in {:5} steps, average reward l'of last {:4} "
                  "episodes is {:8.3f}"
                  .format(row['NTH_EPISODE'], row['TOTAL_REWARD'], row['STEPS_TAKEN'], 100, row['MOVING_AVERAGE_REWARD'])),

    def log(
            self,
            agent_id: int,
            nth_episode: int,
            steps: int,
            total_reward: int,
            moving_average_reward: np.ndarray,
            solved: bool
    ):

        data = [
            agent_id,
            nth_episode,
            steps,
            total_reward,
            moving_average_reward,
            solved
        ]

        log_data = pd.DataFrame([np.array(data)], columns=self.columns)

        self.data_frame = self.data_frame.append(log_data)

        if self.console_log:
            self._episode_log_out(agent_id)

        if self.live_plot:
            self._live_plot_out(agent_id, nth_episode)

        if self.pickle_log and not (nth_episode % self.saving_interval):
            self.pickle_dump(agent_id)

        if self.render_recording:
            if self.frame_buffer:
                imageio.mimwrite(
                    os.path.join('videos_initial_test/', 'agent{}_ep_{}.gif'.format(agent_id, nth_episode)),
                    self.frame_buffer,
                    fps=30
                )
                del self.frame_buffer
                self.frame_buffer = []

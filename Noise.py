#from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import random
import numpy as np
import copy

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Noise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2, open_ai=True):
        """Initialize parameters and noise process."""
        self.open_ai = open_ai
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = seed
        random.seed(seed)
        self.baseline_noise = OUActionNoise(mean=np.zeros(size), std_deviation=sigma * np.ones(size))
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        # self.decay()
        self.state = copy.copy(self.mu)
        self.baseline_noise.reset()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        if self.open_ai:
            return self.baseline_noise()
        return self.state

    def decay(self):
        self.sigma = max(0.35, self.sigma * 0.99)
        self.theta = max(0.15, self.theta * 0.995)

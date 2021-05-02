import numpy as np
import random
import copy

from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.baseline_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(size), sigma=sigma * np.ones(size))
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
        return self.baseline_noise() #self.state
    def decay(self):
        self.sigma = max(0.35, self.sigma*0.99)
        self.theta = max(0.15, self.theta*0.995)
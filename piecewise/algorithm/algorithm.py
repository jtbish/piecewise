import abc
import random

import numpy as np

from piecewise.dtype import Population


class IAlgorithm(metaclass=abc.ABCMeta):
    """Interface for an algorithm."""
    @abc.abstractmethod
    def train_query(self, situation, time_step):
        """Queries the algorithm for an action to perform during training."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_update(self, env_response):
        """Updates and returns the population given the environmental
        response."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_query(self, situation):
        """Queries the algorithm for an action to perform during testing."""
        raise NotImplementedError


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed)


def init_population(max_micros):
    return Population(max_micros)

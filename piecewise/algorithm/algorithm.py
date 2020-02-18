import abc
from collections import namedtuple

from piecewise.dtype import Population
from piecewise.util import ParametrizedMixin

from .hyperparams import get_hyperparam, register_hyperparams
from .rng import seed_rng

AlgorithmResponse = namedtuple("AlgorithmResponse", ["action", "did_explore"])


class AlgorithmABC(ParametrizedMixin, metaclass=abc.ABCMeta):
    """ABC for an algorithm."""
    def __init__(self, hyperparams, seed):
        self.record_parametrization(hyperparams=hyperparams, seed=seed)
        register_hyperparams(hyperparams)
        seed_rng(seed)
        self._population = Population(get_hyperparam("N"))

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

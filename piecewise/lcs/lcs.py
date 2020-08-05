import abc
from collections import namedtuple

from piecewise.dtype import Population

from .hyperparams import get_hyperparam, register_hyperparams
from .rng import seed_rng

LCSTrainResponse = namedtuple("LCSTrainResponse", ["action", "did_explore"])


def setup_meta_params(hyperparams, seed):
    register_hyperparams(hyperparams)
    seed_rng(seed)


class LCS(metaclass=abc.ABCMeta):
    """ABC for an LCS."""
    def __init__(self, rule_repr, population=None):
        self._rule_repr = rule_repr
        self._population = self._init_population(population)

    def _init_population(self, population):
        if population is None:
            return Population(max_micros=get_hyperparam("N"))
        else:
            return population

    @abc.abstractmethod
    def train_query(self, situation, time_step):
        """Queries the algorithm for an action to perform during training."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_update(self, env_response):
        """Updates the population given the environmental response."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_query(self, situation):
        """Queries the algorithm for an action to perform during testing."""
        raise NotImplementedError

    @property
    def rule_repr(self):
        return self._rule_repr

    @property
    def population(self):
        return self._population

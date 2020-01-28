import abc

from piecewise.dtype import Population

from .hyperparams import hyperparams_registry as hps_reg
from .rng import np_random


class AlgorithmABC(metaclass=abc.ABCMeta):
    """ABC for an algorithm."""
    def __init__(self, hyperparams, seed):
        self._register_hyperparams(hyperparams)
        self._seed_np_random_state(seed)
        self._population = self._init_population()

    def _register_hyperparams(self, hyperparams):
        hps_reg.register(hyperparams)

    def _seed_np_random_state(self, seed):
        np_random.seed(seed)

    def _init_population(self):
        return Population(hps_reg["N"])

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

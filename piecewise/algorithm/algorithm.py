import abc

from piecewise.dtype import Population

from .hyperparams import get_hyperparam


class AlgorithmABC(metaclass=abc.ABCMeta):
    """ABC for an algorithm."""
    def __init__(self):
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

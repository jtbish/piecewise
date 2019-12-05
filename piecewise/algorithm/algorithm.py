import abc
import random
from collections import namedtuple

import numpy as np

from piecewise.dtype import Population

CommonComponents = namedtuple("CommonComponents", [
    "matching", "covering", "prediction", "fitness_update", "subsumption",
    "rule_discovery", "deletion"
])


class Algorithm(metaclass=abc.ABCMeta):
    """Abstract base class for an algorithm.

    Responsible for initialising common algorithm components, hyperparameters
    and the population.

    The population is managed by an Algorithm instance and not an LCS instance
    because it is more convenient to have access to the population as an
    instance variable inside the algorithm - many operations need to be
    performed on it using the algorithm components. Not having it accessible as
    an instance variable would require it to be constantly passed as a method
    argument.

    The overarching LCS instance has access to the population at each time step
    via the return value of the Algorithm's train_update method.
    """
    @abc.abstractmethod
    def __init__(self, common_components, rule_repr, hyperparams):
        (self._matching_strat, self._covering_strat, self._prediction_strat,
         self._fitness_update_strat, self._subsumption_strat,
         self._rule_discovery_strat, self._deletion_strat) = common_components

        self._hyperparams = hyperparams
        self._population = Population(max_micros=self._hyperparams["N"])

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    @abc.abstractmethod
    def train_query(self, situation, time_step):
        """Queries the algorithm for an action to perform during training."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_update(self, env_response):
        """Updates and returns the population given the environmental
        feedback."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_query(self, situation):
        """Queries the algorithm for an action to perform during testing."""
        raise NotImplementedError

    def _gen_match_set(self, situation):
        """Returns a match set for the given situation."""
        return self._matching_strat(self._population, situation)

    def _gen_covering_classifier(self, match_set, situation, time_step):
        """Returns a single covering classifier."""
        return self._covering_strat(match_set, situation, time_step)

    def _gen_prediction_array(self, match_set):
        """Generates the prediction array for the given match set."""
        return self._prediction_strat(match_set)

    def _update_fitness(self, operating_set):
        """Performs the fitness update strategy on the given operating set."""
        self._fitness_update_strat(operating_set)

    def _discover_classifiers(self, operating_set, population, situation,
                              time_step):
        """Discovers and inserts new classifiers into the population.

        Any necessary deletion is handled internally by the population.
        """
        return self._rule_discovery_strat(operating_set, population, situation,
                                          time_step)

    def _perform_deletion(self):
        """Performs deletion in the population to keep enforce its capacity
        restriction."""
        self._deletion_strat(self._population)

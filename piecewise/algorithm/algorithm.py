import abc
import random
from collections import namedtuple

import numpy as np

from piecewise.dtype import ClassifierSet, Population

AlgorithmComponents = namedtuple("AlgorithmComponents", [
    "matching", "covering", "prediction", "action_selection",
    "credit_assignment", "fitness_update", "subsumption", "rule_discovery",
    "deletion"
])


class AlgorithmABC(metaclass=abc.ABCMeta):
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
    def __init__(self, components, hyperparams):
        self._init_component_strats(components)
        self._hyperparams = hyperparams
        self._population = Population(max_micros=self._hyperparams["N"])

    def _init_component_strats(self, components):
        self._matching_strat = components.matching
        self._covering_strat = components.covering
        self._prediction_strat = components.prediction
        self._action_selection_strat = components.action_selection
        self._credit_assignment_strat = components.credit_assignment
        self._fitness_update_strat = components.fitness_update
        self._subsumption_strat = components.subsumption
        self._rule_discovery_strat = components.rule_discovery
        self._deletion_strat = components.deletion

    def _set_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)

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

    def _gen_match_set(self, situation):
        return self._matching_strat(self._population, situation)

    def _gen_covering_classifier(self, match_set, situation, time_step):
        return self._covering_strat(match_set, situation, time_step)

    def _gen_prediction_array(self, match_set):
        return self._prediction_strat(match_set)

    def _update_fitness(self, operating_set):
        self._fitness_update_strat(operating_set)

    def _discover_classifiers(self, operating_set, population, situation,
                              time_step):
        return self._rule_discovery_strat(operating_set, population, situation,
                                          time_step)

    def _perform_deletion(self):
        self._deletion_strat(self._population)

    def _gen_action_set(self, match_set, action):
        """GENERATE ACTION SET function from 'An Algorithmic
        Description of XCS' (Butz and Wilson, 2002)."""
        action_set = ClassifierSet()
        for classifier in match_set:
            if classifier.action == action:
                action_set.add(classifier)
        return action_set

    def _select_action(self, prediction_array):
        return self._action_selection_strat(prediction_array)

    def _do_credit_assignment(self, action_set, reward, use_discounting,
                              prediction_array):
        self._credit_assignment_strat(action_set, reward, use_discounting,
                                      prediction_array)

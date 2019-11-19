import abc
from collections import namedtuple

from ..rule_discovery import RuleDiscoveryStrategy

GAOperators = namedtuple("GAOperators", ["selection", "crossover", "mutation"])


class GeneticAlgorithm(RuleDiscoveryStrategy, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, env_action_set, subsumption_strat, hyperparams,
                 ga_operators):
        super().__init__(env_action_set, subsumption_strat, hyperparams)
        (self._selection_strat, self._crossover_strat,
         self._mutation_strat) = ga_operators

    @abc.abstractmethod
    def __call__(self, operating_set, population, situation, time_step):
        raise NotImplementedError

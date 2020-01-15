import abc
from collections import namedtuple


from ..algorithm import Algorithm

ReinforcementComponents = namedtuple("ReinforcementComponents",
                                     ["action_selection", "credit_assignment"])


class ReinforcementAlgorithm(Algorithm, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, common_components, reinforcement_components, rule_repr,
                 hyperparams):
        super().__init__(common_components, rule_repr, hyperparams)
        (self._action_selection_strat,
         self._credit_assignment_strat) = reinforcement_components

    @abc.abstractmethod
    def train_query(self, situation, time_step):
        raise NotImplementedError

    @abc.abstractmethod
    def train_update(self, env_response):
        raise NotImplementedError

    @abc.abstractmethod
    def test_query(self, situation):
        raise NotImplementedError


import abc
import random


class ActionSelectionStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, prediction_array):
        """Returns the next action to perform."""
        raise NotImplementedError


class EpsilonGreedy(ActionSelectionStrategy):
    def __init__(self, prob_explore):
        self._prob_explore = prob_explore

    def __call__(self, prediction_array):
        """SELECT ACTION function from 'An Algorithmic
        Description of XCS' (Butz and Wilson, 2002)."""
        should_exploit = random.random() > self._prob_explore
        if should_exploit:
            return self._select_greedy_action(prediction_array)
        else:
            return self._select_random_action(prediction_array)

    def _select_greedy_action(self, prediction_array):
        possible_sub_array = prediction_array.possible_sub_array()
        return max(possible_sub_array, key=possible_sub_array.get)

    def _select_random_action(self, prediction_array):
        possible_actions_set = prediction_array.possible_actions_set()
        return random.choice(list(possible_actions_set))

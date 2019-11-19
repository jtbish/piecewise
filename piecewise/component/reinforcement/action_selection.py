import abc
import random


class ActionSelectionStrategy(metaclass=abc.ABCMeta):
    def __init__(self, prob_explore):
        self._prob_explore = prob_explore

    @abc.abstractmethod
    def __call__(self, prediction_array):
        """Returns the next action to perform."""
        raise NotImplementedError


class EpsilonGreedy(ActionSelectionStrategy):
    def __init__(self, prob_explore):
        super().__init__(prob_explore)

    def __call__(self, prediction_array):
        """SELECT ACTION function from 'An Algorithmic
        Description of XCS' (Butz and Wilson, 2002)."""
        should_exploit = random.random() > self._prob_explore
        if should_exploit:
            action = max(prediction_array, key=prediction_array.get)
        else:
            possible_actions = prediction_array.keys()
            action = random.choice(tuple(possible_actions))
        return action

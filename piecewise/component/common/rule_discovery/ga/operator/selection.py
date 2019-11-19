import abc
import random


class SelectionStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, hyperparams=None):
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def __call__(self, operating_set):
        """Selects and returns a parent for use in the GA from the operating
        set."""
        raise NotImplementedError


class RouletteWheelSelection(SelectionStrategy):
    def __init__(self):
        super().__init__()

    def __call__(self, operating_set):
        """SELECT OFFSPRING function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        fitness_sum = sum([classifier.fitness for classifier in operating_set])
        choice_point = random.random() * fitness_sum

        fitness_sum = 0
        for classifier in operating_set:
            fitness_sum += classifier.fitness
            if fitness_sum > choice_point:
                return classifier

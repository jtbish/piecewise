import abc
from collections import defaultdict


class PredictionStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, match_set):
        """Returns a prediction array for the given match set."""
        raise NotImplementedError


class FitnessWeightedAvgPrediction(PredictionStrategy):
    def __call__(self, match_set):
        """GENERATE PREDICTION ARRAY function from 'An Algorithmic
        Description of XCS' (Butz and Wilson, 2002)."""
        prediction_array, fitness_sum_array = self._init_arrays()
        self._populate_arrays(prediction_array, fitness_sum_array, match_set)
        self._normalise_prediction_array(prediction_array, fitness_sum_array)
        return prediction_array

    def _init_arrays(self):
        # use default dicts so that 'null' predictions don't appear
        prediction_array = defaultdict(lambda: 0.0)
        fitness_sum_array = defaultdict(lambda: 0.0)
        return prediction_array, fitness_sum_array

    def _populate_arrays(self, prediction_array, fitness_sum_array, match_set):
        for classifier in match_set:
            action = classifier.action
            prediction_array[action] += \
                classifier.prediction * classifier.fitness
            fitness_sum_array[action] += classifier.fitness

    def _normalise_prediction_array(self, prediction_array, fitness_sum_array):
        possible_actions = prediction_array.keys()
        for action in possible_actions:
            if fitness_sum_array[action] != 0:
                prediction_array[action] /= fitness_sum_array[action]

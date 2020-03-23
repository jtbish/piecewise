import logging
from collections import UserDict


class FitnessWeightedAvgPrediction:
    def __init__(self, env_action_set):
        self._env_action_set = env_action_set

    def __call__(self, match_set, situation):
        """GENERATE PREDICTION ARRAY function from 'An Algorithmic
        Description of XCS' (Butz and Wilson, 2002).

        Situation is optional as may or may not be needed depending on
        whether classifiers have constant or computed predictions."""
        self._warn_if_match_set_is_empty(match_set)
        prediction_array, fitness_sum_array = self._init_arrays()
        self._populate_arrays(prediction_array, fitness_sum_array, match_set,
                              situation)
        self._normalise_prediction_array(prediction_array, fitness_sum_array)
        return prediction_array

    def _warn_if_match_set_is_empty(self, match_set):
        match_set_is_empty = match_set.num_micros == 0
        if match_set_is_empty:
            logging.warning("Match set is empty when performing "
                            "prediction.")

    def _init_arrays(self):
        prediction_array = PredictionArray(self._env_action_set)
        fitness_sum_array = {action: 0.0 for action in self._env_action_set}
        return prediction_array, fitness_sum_array

    def _populate_arrays(self, prediction_array, fitness_sum_array, match_set,
                         situation):
        for classifier in match_set:
            action = classifier.action
            prediction = classifier.get_prediction(situation)
            prediction_array[action] += \
                prediction * classifier.fitness
            fitness_sum_array[action] += classifier.fitness

    def _normalise_prediction_array(self, prediction_array, fitness_sum_array):
        possible_actions = prediction_array.keys()
        for action in possible_actions:
            if fitness_sum_array[action] != 0:
                prediction_array[action] /= fitness_sum_array[action]


class PredictionArray(UserDict):
    """Lazy dictionary structure to store predictions."""
    def __init__(self, env_action_set):
        self._env_action_set = env_action_set
        super().__init__()

    @property
    def env_action_set(self):
        return self._env_action_set

    def __getitem__(self, key):
        action = key
        try:
            return self.data[action]
        except KeyError:
            return 0.0

    def __setitem__(self, key, value):
        action = key
        prediction = value
        self.data[action] = prediction

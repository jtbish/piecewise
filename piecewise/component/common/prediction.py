import abc
import random


class PredictionStrategy(metaclass=abc.ABCMeta):
    def __init__(self, env_action_set):
        self._env_action_set = env_action_set

    @abc.abstractmethod
    def __call__(self, match_set):
        """Returns a prediction array for the given match set."""
        raise NotImplementedError


class FitnessWeightedAvgPrediction(PredictionStrategy):
    def __init__(self, env_action_set):
        super().__init__(env_action_set)

    def __call__(self, match_set):
        """GENERATE PREDICTION ARRAY function from 'An Algorithmic
        Description of XCS' (Butz and Wilson, 2002)."""
        prediction_array, fitness_sum_array = self._init_arrays()
        self._populate_arrays(prediction_array, fitness_sum_array, match_set)
        self._normalise_prediction_array(prediction_array, fitness_sum_array)
        return prediction_array

    def _init_arrays(self):
        prediction_array = PredictionArray(self._env_action_set)
        fitness_sum_array = {action: 0.0 for action in self._env_action_set}
        return prediction_array, fitness_sum_array

    def _populate_arrays(self, prediction_array, fitness_sum_array, match_set):
        match_set_is_empty = match_set.num_micros == 0
        if match_set_is_empty:
            print("WARNING: match set is empty when doing prediction.")

        for classifier in match_set:
            action = classifier.action
            prediction_array[action] += \
                classifier.prediction * classifier.fitness
            fitness_sum_array[action] += classifier.fitness

    def _normalise_prediction_array(self, prediction_array, fitness_sum_array):
        possible_actions = prediction_array.actions()
        for action in possible_actions:
            if fitness_sum_array[action] != 0:
                prediction_array[action] /= fitness_sum_array[action]


class PredictionArray:
    """Structure to nicely encapsulate null action predictions and selecting
    actions from caller."""
    def __init__(self, env_action_set):
        self._arr = {action: None for action in env_action_set}

    def __getitem__(self, key):
        action = key
        prediction = self._arr[action]
        prediction = \
            self._lazily_make_prediction_non_null(prediction)
        return prediction

    def _lazily_make_prediction_non_null(self, prediction):
        if prediction is None:
            prediction = 0.0
        return prediction

    def __setitem__(self, key, value):
        action = key
        prediction = value
        self._arr[action] = prediction

    def random_action(self):
        return random.choice(self._get_possible_actions)

    def _get_possible_actions(self):
        return [
            action for (action, prediction) in self._arr.items()
            if prediction is not None
        ]

    def greedy_action(self):
        possible_sub_arr = self._get_possible_sub_arr()
        return max(possible_sub_arr, key=possible_sub_arr.get)

    def _get_possible_sub_arr(self):
        return {
            action: prediction for (action, prediction) in self._arr.items()
            if prediction is not None
        }

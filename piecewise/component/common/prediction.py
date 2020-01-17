import abc


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
            # TODO change to logging call
            print("WARNING: match set is empty when doing prediction.")

        for classifier in match_set:
            action = classifier.action
            prediction_array[action] += \
                classifier.prediction * classifier.fitness
            fitness_sum_array[action] += classifier.fitness

    def _normalise_prediction_array(self, prediction_array, fitness_sum_array):
        for action in prediction_array.possible_actions_set():
            if fitness_sum_array[action] != 0:
                prediction_array[action] /= fitness_sum_array[action]


class PredictionArray:
    """Data structure to nicely encapsulate null action predictions from
    client.

    The main idea behind the data structure is to store prediction values as
    null (None) to begin with, and lazily make them zero as necessary when
    the client code requests them via key indexing.
    """
    def __init__(self, env_action_set):
        self._arr = {action: None for action in env_action_set}

    def __getitem__(self, key):
        action = key
        prediction = self._arr[action]
        prediction = \
            self._make_prediction_non_null_if_needed(prediction)
        return prediction

    def _make_prediction_non_null_if_needed(self, prediction):
        if prediction is None:
            prediction = 0.0
        return prediction

    def __setitem__(self, key, value):
        action = key
        prediction = value
        self._arr[action] = prediction

    def possible_actions_set(self):
        """Returns the set of actions with non-null predictions.

        Used with action selection strategy."""
        return {
            action
            for (action, prediction) in self._arr.items()
            if prediction is not None
        }

    def possible_sub_array(self):
        """Returns the sub-array with non-null predictions.

        Used with action selection strategy."""
        return {
            action: prediction
            for (action, prediction) in self._arr.items()
            if prediction is not None
        }

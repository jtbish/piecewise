import logging
import sys


class ExperienceWeightedAvgError:
    def __init__(self, env_action_set):
        self._env_action_set = env_action_set

    def __call__(self, match_set):
        self._warn_if_match_set_is_empty(match_set)
        return self._gen_weighted_mean_error_array(match_set)

    def _warn_if_match_set_is_empty(self, match_set):
        match_set_is_empty = match_set.num_micros == 0
        if match_set_is_empty:
            logging.warning("Match set is empty when performing "
                            "prediction.")

    def _gen_weighted_mean_error_array(self, match_set):
        # experience represents number of updates so use it as a weighting
        # factor
        numerators = {action: 0.0 for action in self._env_action_set}
        denominators = {action: 0 for action in self._env_action_set}
        for classifier in match_set:
            action = classifier.action
            numerator[action] += classifier.experience * classifier.error
            denominator[action] += classifier.experience
        for action in self._env_action_set:
            if denominator[action] == 0:
                # no information so be pessimistic
                max_float = sys.float_info.max
                weighted_mean_error_array[action] = max_float
            else:
                weighted_mean_error_array[action] = numerator/denominator
        return weighted_mean_error_array

class NullErrorCalculation:

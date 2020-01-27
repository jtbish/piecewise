import logging
import random


class EpsilonGreedy:
    def __init__(self, hyperparams):
        self._prob_explore = hyperparams["p_explore"]

    def __call__(self, prediction_array):
        """SELECT ACTION function from 'An Algorithmic
        Description of XCS' (Butz and Wilson, 2002)."""
        should_exploit = random.random() > self._prob_explore
        if should_exploit:
            return select_greedy_action(prediction_array)
        else:
            return _select_random_action_with_valid_prediction(
                prediction_array)


def select_greedy_action(prediction_array):
    if len(prediction_array) != 0:
        return max(prediction_array, key=prediction_array.get)
    else:
        return _fallback_to_random_selection_from_action_set(
            prediction_array.env_action_set)


def _select_random_action_with_valid_prediction(prediction_array):
    if len(prediction_array) != 0:
        possible_actions = prediction_array.keys()
        return random.choice(list(possible_actions))
    else:
        return _fallback_to_random_selection_from_action_set(
            prediction_array.env_action_set)


def _fallback_to_random_selection_from_action_set(env_action_set):
    logging.warning("Falling back to random action selection due to empty "
                    "prediction array.")
    return random.choice(list(env_action_set))

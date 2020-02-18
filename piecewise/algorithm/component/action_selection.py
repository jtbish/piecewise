import logging
from collections import namedtuple

from piecewise.algorithm.hyperparams import get_hyperparam
from piecewise.algorithm.rng import get_rng

ActionSelectResponse = namedtuple("ActionSelectResponse",
                                  ["action", "did_explore"])


class EpsilonGreedy:
    def __call__(self, prediction_array):
        """SELECT ACTION function from 'An Algorithmic
        Description of XCS' (Butz and Wilson, 2002)."""
        should_explore = get_rng().rand() <= get_hyperparam("p_explore")
        if should_explore:
            action = \
                _select_random_action_with_valid_prediction(prediction_array)
        else:
            action = select_greedy_action(prediction_array)
        return ActionSelectResponse(action=action, did_explore=should_explore)


def select_greedy_action(prediction_array):
    if len(prediction_array) != 0:
        return max(prediction_array, key=prediction_array.get)
    else:
        return _fallback_to_random_selection_from_action_set(
            prediction_array.env_action_set)


def _select_random_action_with_valid_prediction(prediction_array):
    if len(prediction_array) != 0:
        possible_actions = prediction_array.keys()
        return get_rng().choice(list(possible_actions))
    else:
        return _fallback_to_random_selection_from_action_set(
            prediction_array.env_action_set)


def _fallback_to_random_selection_from_action_set(env_action_set):
    logging.warning("Falling back to random action selection due to empty "
                    "prediction array.")
    return get_rng().choice(list(env_action_set))

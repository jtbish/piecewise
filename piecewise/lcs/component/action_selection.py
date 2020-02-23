import logging
from collections import namedtuple

from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng

ActionSelectResponse = namedtuple("ActionSelectResponse",
                                  ["action", "did_explore"])


class FixedEpsilonGreedy:
    def __call__(self, prediction_array):
        """SELECT ACTION function from 'An Algorithmic
        Description of XCS' (Butz and Wilson, 2002)."""
        epsilon = get_hyperparam("p_explore")
        return _epsilon_greedy(prediction_array, epsilon)


class DecayingEpsilonGreedy:
    def __init__(self):
        self._epsilon = 1.0

    def __call__(self, prediction_array):
        self._apply_decay()
        logging.debug(f"Epsilon = {self._epsilon}")
        return _epsilon_greedy(prediction_array, self._epsilon)

    def _apply_decay(self):
        self._epsilon *= get_hyperparam("e_greedy_decay_factor")


def _epsilon_greedy(prediction_array, epsilon):
    should_explore = get_rng().rand() <= epsilon
    if should_explore:
        action = \
            _select_random_action_with_valid_prediction(prediction_array)
    else:
        action = select_greedy_action(prediction_array)
    return ActionSelectResponse(action=action, did_explore=should_explore)


def _select_random_action_with_valid_prediction(prediction_array):
    if len(prediction_array) != 0:
        possible_actions = prediction_array.keys()
        return get_rng().choice(list(possible_actions))
    else:
        return _fallback_to_random_selection_from_action_set(
            prediction_array.env_action_set)


def select_greedy_action(prediction_array):
    if len(prediction_array) != 0:
        return max(prediction_array, key=prediction_array.get)
    else:
        return _fallback_to_random_selection_from_action_set(
            prediction_array.env_action_set)


def _fallback_to_random_selection_from_action_set(env_action_set):
    logging.warning("Falling back to random action selection due to empty "
                    "prediction array.")
    return get_rng().choice(list(env_action_set))

import logging
from collections import namedtuple

from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng

ActionSelectResponse = namedtuple("ActionSelectResponse",
                                  ["action", "did_explore"])

class GreedyActionSelection:
    def __call__(self, prediction_array, time_step=None):
        action = select_greedy_action(prediction_array)
        return ActionSelectResponse(action, did_explore=False)


class FixedEpsilonGreedy:
    def __call__(self, prediction_array, time_step=None):
        """SELECT ACTION function from 'An Algorithmic
        Description of XCS' (Butz and Wilson, 2002)."""
        epsilon = get_hyperparam("p_explore")
        return _epsilon_greedy(prediction_array, epsilon)


class LinearDecayEpsilonGreedy:
    def __init__(self):
        self._epsilon_max = 1.0
        self._epsilon = self._epsilon_max

    def __call__(self, prediction_array, time_step):
        self._decay_epsilon(time_step)
        return _epsilon_greedy(prediction_array, self._epsilon)

    def _decay_epsilon(self, time_step):
        decayed_val = self._epsilon_max - \
            get_hyperparam("e_greedy_decay_factor")*time_step
        self._epsilon = max(decayed_val, get_hyperparam("e_greedy_min_epsilon"))


def _epsilon_greedy(prediction_array, epsilon):
    logging.debug(f"Epsilon = {epsilon}")
    assert 0.0 <= epsilon <= 1.0
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

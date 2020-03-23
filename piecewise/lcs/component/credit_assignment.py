import numpy as np
import math

from piecewise.lcs.hyperparams import get_hyperparam


def _update_action_set_size(classifier, action_set):
    action_set_size_diff = action_set.num_micros \
            - classifier.action_set_size

    if classifier.experience < _num_initial_adjust_steps():
        classifier.action_set_size += action_set_size_diff / \
            classifier.experience
    else:
        classifier.action_set_size += \
            get_hyperparam("beta") * action_set_size_diff


def _num_initial_adjust_steps():
    return 1 / get_hyperparam("beta")


class XCSCreditAssignment:
    def __call__(self, action_set, payoff, situation=None):
        """UPDATE SET function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        for classifier in action_set:
            classifier.experience += 1
            payoff_diff = payoff - classifier.get_prediction()
            self._update_prediction(classifier, payoff_diff)
            self._update_prediction_error(classifier, payoff_diff)
            _update_action_set_size(classifier, action_set)

    def _update_prediction(self, classifier, payoff_diff):
        if classifier.experience < _num_initial_adjust_steps():
            updated_prediction = classifier.get_prediction() +  \
                payoff_diff/classifier.experience
        else:
            updated_prediction = classifier.get_prediction() + \
                get_hyperparam("beta") * payoff_diff
        classifier.set_prediction(updated_prediction)

    def _update_prediction_error(self, classifier, payoff_diff):
        error_diff = abs(payoff_diff) - classifier.error

        if classifier.experience < _num_initial_adjust_steps():
            classifier.error += error_diff / classifier.experience
        else:
            classifier.error += get_hyperparam("beta") * error_diff


class XCSFLinearPredictionCreditAssignment:
    def __call__(self, action_set, payoff, situation):
        for classifier in action_set:
            classifier.experience += 1
            payoff_diff = payoff - classifier.get_prediction(situation)
            self._update_weight_vec(classifier, payoff_diff)
            self._update_prediction_error(classifier, payoff_diff)
            _update_action_set_size(classifier, action_set)

    def _update_weight_vec(self, classifier, situation, payoff_diff):
        weight_deltas = self._calc_weight_deltas(situation, payoff_diff)
        self._apply_weight_deltas(classifier, weight_deltas)

    def _calc_weight_deltas(self, situation, payoff_diff):
        augmented_situation = self._prepend_threshold_to_situation(situation)
        situation_norm = math.sqrt(
            sum([elem**2 for elem in augmented_situation]))  # L2 norm
        weight_deltas = []
        for elem in augmented_situation:
            delta = (get_hyperparam("eta") /
                     situation_norm) * payoff_diff * elem
            weight_deltas.append(delta)
        return weight_deltas

    def _prepend_threshold_to_situation(self, situation):
        return np.insert(situation, 0, get_hyperparam("x_nought"))

    def _apply_weight_deltas(self, classifier, weight_deltas):
        for idx, delta in enumerate(weight_deltas):
            classifier.weight_vec[idx] += delta

    def _update_prediction_error(self, classifier, payoff_diff):
        error_diff = abs(payoff_diff) - classifier.error
        classifier.error += get_hyperparam("beta") * error_diff

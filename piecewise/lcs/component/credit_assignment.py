import numpy as np

from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.util.classifier_set_stats import calc_summary_stat


def update_action_set_size(classifier, action_set):
    action_set_size_diff = action_set.num_micros \
            - classifier.action_set_size

    if classifier.experience < (1 / get_hyperparam("beta")):
        classifier.action_set_size += action_set_size_diff / \
            classifier.experience
    else:
        classifier.action_set_size += \
            get_hyperparam("beta") * action_set_size_diff


class XCSCreditAssignment:
    def __call__(self, action_set, payoff, situation=None):
        """UPDATE SET function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        for classifier in action_set:
            classifier.experience += 1
            payoff_diff = payoff - classifier.get_prediction()
            self._update_prediction(classifier, payoff_diff)
            self._update_prediction_error(classifier, payoff_diff)
            update_action_set_size(classifier, action_set)

    def _update_prediction(self, classifier, payoff_diff):
        if classifier.experience < (1 / get_hyperparam("beta")):
            updated_prediction = classifier.get_prediction() +  \
                payoff_diff/classifier.experience
        else:
            updated_prediction = classifier.get_prediction() + \
                get_hyperparam("beta") * payoff_diff
        classifier.set_prediction(updated_prediction)

    def _update_prediction_error(self, classifier, payoff_diff):
        error_diff = abs(payoff_diff) - classifier.error
        if classifier.experience < (1 / get_hyperparam("beta")):
            classifier.error += error_diff / classifier.experience
        else:
            classifier.error += get_hyperparam("beta") * error_diff


class XCSFLinearPredictionCreditAssignment:
    def __call__(self, action_set, payoff, situation):
        niche_min_error = calc_summary_stat(action_set, "min", "error")
        for classifier in action_set:
            classifier.experience += 1
            payoff_diff = payoff - classifier.get_prediction(situation)
            self._update_weight_vec(classifier, payoff_diff, situation)
            self._update_niche_min_error(classifier, niche_min_error)
            self._update_prediction_error(classifier, payoff_diff)
            update_action_set_size(classifier, action_set)

    def _update_weight_vec(self, classifier, payoff_diff, situation):
        weight_deltas = self._calc_weight_deltas(payoff_diff, situation)
        self._apply_weight_deltas(classifier, weight_deltas)

    def _calc_weight_deltas(self, payoff_diff, situation):
        augmented_situation = self._prepend_threshold_to_situation(situation)
        # Normalise by squared L2 norm: see e.g.
        # https://danieltakeshi.github.io/2015-07-29-the-least-mean-squares-algorithm/
        normalisation_term = sum([elem**2 for elem in augmented_situation])
        weight_deltas = []
        for elem in augmented_situation:
            delta = (get_hyperparam("eta") / normalisation_term) \
                * payoff_diff * elem
            weight_deltas.append(delta)
        return weight_deltas

    def _prepend_threshold_to_situation(self, situation):
        return np.insert(situation, 0, get_hyperparam("x_nought"))

    def _apply_weight_deltas(self, classifier, weight_deltas):
        for idx, delta in enumerate(weight_deltas):
            classifier.weight_vec[idx] += delta

    def _update_niche_min_error(self, classifier, niche_min_error):
        # Use MAM for mu param
        niche_min_error_diff = niche_min_error - classifier.niche_min_error
        if classifier.experience < (1 / get_hyperparam("beta_e")):
            classifier.niche_min_error += \
                niche_min_error_diff / classifier.experience
        else:
            classifier.niche_min_error += \
                get_hyperparam("beta_e") * niche_min_error_diff

    def _update_prediction_error(self, classifier, payoff_diff):
#        first_term = abs(payoff_diff) - classifier.niche_min_error
#        if first_term < 0:
#            first_term = get_hyperparam("epsilon_nought")
#        error_diff = first_term - classifier.error
        error_diff = abs(payoff_diff) - classifier.error

        # Use MAM for error
        if classifier.experience < (1 / get_hyperparam("beta")):
            classifier.error += error_diff / classifier.experience
        else:
            classifier.error += get_hyperparam("beta") * error_diff

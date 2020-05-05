import numpy as np

from piecewise.lcs.component.credit_assignment import update_action_set_size
from piecewise.lcs.hyperparams import get_hyperparam


class FuzzyXCSFLinearPredictionCreditAssignment:
    def __call__(self, action_set, payoff, situation):
        total_matching_degrees = sum(
            [classifier.matching_degree for classifier in action_set])
        for classifier in action_set:
            credit_weight = (classifier.matching_degree /
                             total_matching_degrees)
            self._update_experience(classifier, credit_weight)
            payoff_diff = payoff - classifier.get_prediction(situation)
            self._update_weight_vec(classifier, payoff_diff, situation,
                                    credit_weight)
            self._update_prediction_error(classifier, payoff_diff,
                                          credit_weight)
            self._update_action_set_size(classifier, action_set)

    def _update_experience(self, classifier, credit_weight):
        classifier.experience += credit_weight

    def _update_weight_vec(self, classifier, payoff_diff, situation,
                           credit_weight):
        weight_deltas = self._calc_weight_deltas(payoff_diff, situation,
                                                 credit_weight)
        self._apply_weight_deltas(classifier, weight_deltas)

    def _calc_weight_deltas(self, payoff_diff, situation, credit_weight):
        augmented_situation = self._prepend_threshold_to_situation(situation)
        # Normalise by squared L2 norm: see e.g.
        # https://danieltakeshi.github.io/2015-07-29-the-least-mean-squares-algorithm/
        normalisation_term = sum([elem**2 for elem in augmented_situation])
        weight_deltas = []
        for elem in augmented_situation:
            delta = (get_hyperparam("eta") / normalisation_term) \
                * payoff_diff * elem
            weight_deltas.append(credit_weight * delta)
        return weight_deltas

    def _prepend_threshold_to_situation(self, situation):
        return np.insert(situation, 0, get_hyperparam("x_nought"))

    def _apply_weight_deltas(self, classifier, weight_deltas):
        for idx, delta in enumerate(weight_deltas):
            classifier.weight_vec[idx] += delta

    def _update_prediction_error(self, classifier, payoff_diff, credit_weight):
        error_diff = abs(payoff_diff) - classifier.error
        # Use MAM for error
#        if classifier.experience < (1 / get_hyperparam("beta")):
#            # TODO credit weight here?
#            classifier.error += error_diff / classifier.experience
#        else:
        classifier.error += \
            (credit_weight * get_hyperparam("beta") * error_diff)

    def _update_action_set_size(self, classifier, action_set):
        action_set_size_diff = action_set.num_micros \
                - classifier.action_set_size
        classifier.action_set_size += \
            get_hyperparam("beta") * action_set_size_diff

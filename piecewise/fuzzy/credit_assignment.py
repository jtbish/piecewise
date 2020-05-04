import numpy as np

from piecewise.lcs.component.credit_assignment import update_action_set_size
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.util.classifier_set_stats import calc_summary_stat


class FuzzyXCSFLinearPredictionCreditAssignment:
    def __call__(self, action_set, payoff, situation):
        niche_min_error = calc_summary_stat(action_set, "min", "error")
        total_matching_degrees = sum(
            [classifier.matching_degree for classifier in action_set])
        for classifier in action_set:
            credit_weight = (classifier.matching_degree /
                             total_matching_degrees)
            self._update_experience(classifier, credit_weight)
            payoff_diff = payoff - classifier.get_prediction(situation)
            self._update_weight_vec(classifier, payoff_diff, situation,
                                    credit_weight)
            self._update_niche_min_error(classifier, niche_min_error,
                                         credit_weight)
            self._update_prediction_error(classifier, payoff_diff,
                                          credit_weight)
            update_action_set_size(classifier, action_set)

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

    def _update_niche_min_error(self, classifier, niche_min_error,
                                credit_weight):
        # Use MAM for mu param
        niche_min_error_diff = niche_min_error - classifier.niche_min_error
        if classifier.experience < (1 / get_hyperparam("beta_e")):
            # TODO credit weight here?
            classifier.niche_min_error += \
                niche_min_error_diff / classifier.experience
        else:
            classifier.niche_min_error += \
                (credit_weight * get_hyperparam("beta_e") *
                    niche_min_error_diff)

    def _update_prediction_error(self, classifier, payoff_diff, credit_weight):
        #        first_term = abs(payoff_diff) - classifier.niche_min_error
        #        if first_term < 0:
        #            first_term = get_hyperparam("epsilon_nought")
        #        error_diff = first_term - classifier.error
        error_diff = abs(payoff_diff) - classifier.error

        # Use MAM for error
        if classifier.experience < (1 / get_hyperparam("beta")):
            # TODO credit weight here?
            classifier.error += error_diff / classifier.experience
        else:
            classifier.error += \
                (credit_weight * get_hyperparam("beta") * error_diff)

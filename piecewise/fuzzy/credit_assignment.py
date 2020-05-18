import numpy as np

from piecewise.lcs.hyperparams import get_hyperparam


class FuzzyXCSFLinearPredictionCreditAssignment:
    def __init__(self, rule_repr):
        self._rule_repr = rule_repr

    def __call__(self, action_set, payoff, situation):
        matching_degrees = [
            classifier.calc_matching_degree(self._rule_repr, situation)
            for classifier in action_set
        ]
        total_matching_degrees = sum(matching_degrees)
        assert total_matching_degrees > 0.0
        for (classifier, matching_degree) in zip(action_set, matching_degrees):
            credit_weight = (matching_degree / total_matching_degrees)
            assert credit_weight > 0.0
            self._update_experience(classifier, credit_weight)
            payoff_diff = payoff - classifier.get_prediction(situation)
            self._update_prediction(classifier, payoff_diff, situation,
                    credit_weight)
            self._update_prediction_error(classifier, payoff_diff,
                                          credit_weight)
            self._update_action_set_size(classifier, action_set)

    def _update_experience(self, classifier, credit_weight):
        classifier.experience += credit_weight

    def _update_prediction(self, classifier, payoff_diff, situation,
            credit_weight):
        # Weighted recursive least squares
        enriched_situation = self._prepend_threshold_to_situation(situation)

        # calc cov mat update rate - scalar
        beta_rls = 1 + credit_weight* \
            (np.dot(enriched_situation, classifier.cov_mat).dot(enriched_situation))

        # update cov mat
        classifier.cov_mat -= (1/beta_rls)*credit_weight* \
            (np.dot(classifier.cov_mat,
                enriched_situation).dot(enriched_situation).dot(
                    classifier.cov_mat))

        # calc gain vector for weights
        gain_vec = np.dot(classifier.cov_mat, enriched_situation)

        # update weights with gain vec and payoff diff (error)
        assert len(gain_vec) == len(classifier.weight_vec)
        for idx, gain in enumerate(gain_vec)
            classifier.weight_vec[idx] += gain*credit_weight*payoff_diff

    def _prepend_threshold_to_situation(self, situation):
        return np.insert(situation, 0, get_hyperparam("x_nought"))

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
        classifier.action_set_size += \ get_hyperparam("beta") * action_set_size_diff

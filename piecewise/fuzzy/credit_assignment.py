import numpy as np
import logging

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
            logging.debug(f"{classifier}, matching degree {matching_degree:.4f}")
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

        should_reset_cov_mat = \
            (classifier.experience - classifier.cov_mat_reset_stamp) \
                >= get_hyperparam("tau_rls")
        if should_reset_cov_mat:
            logging.debug("Resetting clfr cov mat")
            classifier.reset_cov_mat(get_hyperparam("delta_rls"))

        # calc cov mat update rate
        beta_rls = 1 + credit_weight* \
            np.asscalar((np.dot(enriched_situation,
                classifier.cov_mat)).dot(enriched_situation.transpose()))

        # update cov mat
        classifier.cov_mat -= (1/beta_rls)*credit_weight* \
            (np.dot(classifier.cov_mat, enriched_situation.transpose())).dot(
                (np.dot(enriched_situation, classifier.cov_mat)))

        # calc gain vector for weights
        gain_vec = np.dot(classifier.cov_mat, enriched_situation.transpose())
        gain_vec = gain_vec.flatten()

        # update weights with gain vec and payoff diff (error)
        assert len(gain_vec) == len(classifier.weight_vec)
        for idx, gain in enumerate(gain_vec):
            classifier.weight_vec[idx] += gain*credit_weight*payoff_diff

    def _prepend_threshold_to_situation(self, situation):
        # return 1x(d+1) row vector
        res = np.insert(situation, 0, get_hyperparam("x_nought"))
        res = np.reshape(res, (1, len(res)))
        return res

    def _update_prediction_error(self, classifier, payoff_diff, credit_weight):
        error_diff = abs(payoff_diff) - classifier.error
        classifier.error += \
            (credit_weight * get_hyperparam("beta") * error_diff)

    def _update_action_set_size(self, classifier, action_set):
        action_set_size_diff = action_set.num_micros \
            - classifier.action_set_size
        classifier.action_set_size += get_hyperparam("beta") * action_set_size_diff

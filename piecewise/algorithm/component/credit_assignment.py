from piecewise.algorithm.hyperparams import hyperparams_registry as hps_reg


class XCSCreditAssignment:
    def __call__(self, action_set, payoff):
        """UPDATE SET function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        for classifier in action_set:
            classifier.experience += 1
            payoff_diff = payoff - classifier.prediction
            self._update_prediction(classifier, payoff_diff)
            self._update_prediction_error(classifier, payoff_diff)
            self._update_action_set_size(classifier, action_set)

    def _update_prediction(self, classifier, payoff_diff):
        if classifier.experience < self._num_initial_adjust_steps():
            classifier.prediction += \
                payoff_diff/classifier.experience
        else:
            classifier.prediction += hps_reg["beta"] * \
                payoff_diff

    def _update_prediction_error(self, classifier, payoff_diff):
        error_diff = abs(payoff_diff) - classifier.error

        if classifier.experience < self._num_initial_adjust_steps():
            classifier.error += error_diff / classifier.experience
        else:
            classifier.error += hps_reg["beta"] * error_diff

    def _update_action_set_size(self, classifier, action_set):
        action_set_size_diff = action_set.num_micros \
                - classifier.action_set_size

        if classifier.experience < self._num_initial_adjust_steps():
            classifier.action_set_size += action_set_size_diff / \
                classifier.experience
        else:
            classifier.action_set_size += \
                hps_reg["beta"] * action_set_size_diff

    def _num_initial_adjust_steps(self):
        return 1 / hps_reg["beta"]

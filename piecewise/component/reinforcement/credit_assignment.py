import abc


class CreditAssignmentStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, hyperparams=None):
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def __call__(self, action_set, reward, use_discounting, prediction_array):
        """Assigns credit to classifiers in the action set, using information
        from the other parameters."""
        raise NotImplementedError


class XCSCreditAssignment(CreditAssignmentStrategy):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self._num_initial_adjust_steps = 1 / self._hyperparams["beta"]

    def __call__(self, action_set, reward, use_discounting, prediction_array):
        """UPDATE SET function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002), with internal payoff calculation instead of
        payoff given as a parameter."""
        payoff = self._calc_payoff(reward, use_discounting, prediction_array)

        for classifier in action_set:
            classifier.experience += 1
            payoff_diff = payoff - classifier.prediction
            self._update_prediction(classifier, payoff_diff)
            self._update_prediction_error(classifier, payoff_diff)
            self._update_action_set_size(classifier, action_set)

    def _calc_payoff(self, reward, use_discounting, prediction_array):
        if use_discounting:
            # Q-learning style payoff calculation
            max_prediction = max(prediction_array.values())
            payoff = reward + (self._hyperparams["gamma"] * max_prediction)
        else:
            payoff = reward

        return payoff

    def _update_prediction(self, classifier, payoff_diff):
        if classifier.experience < self._num_initial_adjust_steps:
            classifier.prediction += \
                payoff_diff/classifier.experience
        else:
            classifier.prediction += self._hyperparams["beta"] * \
                payoff_diff

    def _update_prediction_error(self, classifier, payoff_diff):
        error_diff = abs(payoff_diff) - classifier.error

        if classifier.experience < self._num_initial_adjust_steps:
            classifier.error += error_diff / classifier.experience
        else:
            classifier.error += self._hyperparams["beta"] * error_diff

    def _update_action_set_size(self, classifier, action_set):
        action_set_size_diff = action_set.num_micros \
                - classifier.action_set_size

        if classifier.experience < self._num_initial_adjust_steps:
            classifier.action_set_size += action_set_size_diff / \
                classifier.experience
        else:
            classifier.action_set_size += \
                self._hyperparams["beta"] * action_set_size_diff

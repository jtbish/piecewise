from piecewise.algorithm.hyperparams import hyperparams_registry as hps_reg


class XCSAccuracyFitnessUpdate:
    _MAX_ACCURACY = 1

    def __call__(self, action_set):
        """UPDATE FITNESS function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002).
        """
        accuracy_vec, accuracy_sum = \
            self._calc_accuracy_vec_and_accuracy_sum(action_set)
        self._update_fitness_values(action_set, accuracy_vec, accuracy_sum)

    def _calc_accuracy_vec_and_accuracy_sum(self, action_set):
        accuracy_sum = 0
        accuracy_vec = []
        for classifier in action_set:
            is_below_error_threshold = classifier.error < \
                hps_reg["epsilon_nought"]
            if is_below_error_threshold:
                accuracy = self._MAX_ACCURACY
            else:
                accuracy = self._calc_accuracy(classifier)
            accuracy_vec.append(accuracy)
            accuracy_sum += accuracy * classifier.numerosity
        return accuracy_vec, accuracy_sum

    def _calc_accuracy(self, classifier):
        return hps_reg["alpha"] * \
                (classifier.error / hps_reg["epsilon_nought"])\
                ** (-1 * hps_reg["nu"])

    def _update_fitness_values(self, action_set, accuracy_vec, accuracy_sum):
        for (classifier, accuracy) in zip(action_set, accuracy_vec):
            adjustment = \
                ((accuracy*classifier.numerosity/accuracy_sum) -
                 classifier.fitness)
            classifier.fitness += hps_reg["beta"] * adjustment

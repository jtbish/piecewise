class XCSAccuracyFitnessUpdate:
    _MAX_ACCURACY = 1

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams

    def __call__(self, operating_set):
        """UPDATE FITNESS function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002).
        """
        action_set = operating_set
        accuracy_vec, accuracy_sum = \
            self._calc_accuracy_vec_and_accuracy_sum(action_set)
        self._update_fitness_values(action_set, accuracy_vec, accuracy_sum)

    def _calc_accuracy_vec_and_accuracy_sum(self, action_set):
        accuracy_sum = 0
        accuracy_vec = []
        for classifier in action_set:
            is_below_error_threshold = classifier.error < \
                self._hyperparams["epsilon_nought"]
            if is_below_error_threshold:
                accuracy = self._MAX_ACCURACY
            else:
                accuracy = self._calc_accuracy(classifier)
            accuracy_vec.append(accuracy)
            accuracy_sum += accuracy * classifier.numerosity
        return accuracy_vec, accuracy_sum

    def _calc_accuracy(self, classifier):
        return self._hyperparams["alpha"] * \
                (classifier.error / self._hyperparams["epsilon_nought"])\
                ** (-self._hyperparams["nu"])

    def _update_fitness_values(self, action_set, accuracy_vec, accuracy_sum):
        for (classifier, accuracy) in zip(action_set, accuracy_vec):
            adjustment = \
                ((accuracy*classifier.numerosity/accuracy_sum) -
                 classifier.fitness)
            classifier.fitness += self._hyperparams["beta"] * adjustment

import math

from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng


class RouletteWheelSelection:
    def __call__(self, operating_set):
        """SELECT OFFSPRING function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        fitness_sum = sum([classifier.fitness for classifier in operating_set])
        choice_point = get_rng().rand() * fitness_sum

        fitness_sum = 0
        for classifier in operating_set:
            fitness_sum += classifier.fitness
            if fitness_sum > choice_point:
                return classifier


class TournamentSelection:
    def __call__(self, operating_set):
        tournament_size = math.ceil(
            get_hyperparam("tau") * operating_set.num_macros)
        assert 1 <= tournament_size <= operating_set.num_macros
        best_classifier = \
            self._select_random_classifier_from_set(operating_set)
        for _ in range(2, (tournament_size + 1)):
            classifier = self._select_random_classifier_from_set(operating_set)
            if classifier.fitness > best_classifier.fitness:
                best_classifier = classifier
        return best_classifier

    def _select_random_classifier_from_set(self, operating_set):
        return get_rng().choice(list(operating_set))

import abc
import random

from piecewise.util.classifier_set_stats import calc_summary_stat


class DeletionStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, hyperparams=None):
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def __call__(self, population):
        """Selects a single classifier to delete from the population.

        Precondition: This method should only ever be called if the population
        is at or past its microclassifier capacity."""
        return NotImplementedError


class XCSRouletteWheelDeletion(DeletionStrategy):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    def __call__(self, population):
        """First loop (selecting classifier to 'delete') of
        DELETE FROM POPULATION function from 'An Algorithmic Description of
        XCS' (Butz and Wilson, 2002)."""
        votes = [
            self._calc_deletion_vote(classifier, population)
            for classifier in population
        ]
        vote_sum = sum(votes)
        choice_point = random.random() * vote_sum

        vote_sum = 0
        for (classifier, vote) in zip(population, votes):
            vote_sum += vote
            if vote_sum > choice_point:
                return classifier

    def _calc_deletion_vote(self, classifier, population):
        """DELETION VOTE function from 'An Algorithmic Description of
        XCS' (Butz and Wilson, 2002)."""
        vote = classifier.action_set_size * classifier.numerosity
        mean_fitness_in_pop = calc_summary_stat(population, "mean", "fitness")
        fitness_numerosity_ratio = classifier.fitness / classifier.numerosity

        has_sufficient_experience = classifier.experience > \
            self._hyperparams["theta_del"]
        has_low_fitness = fitness_numerosity_ratio < \
            (self._hyperparams["delta"] * mean_fitness_in_pop)
        if has_sufficient_experience and has_low_fitness:
            vote *= mean_fitness_in_pop / fitness_numerosity_ratio

        return vote

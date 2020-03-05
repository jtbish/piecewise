import abc

from piecewise.lcs.hyperparams import get_hyperparam


class ISubsumptionStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def does_subsume(self, subsumer_classifier, subsumee_classifier):
        """Determines if the subsumer classifier subsumes the subsumee
        classifier."""
        raise NotImplementedError

    @abc.abstractmethod
    def could_subsume(self, classifier):
        """Determines if the given classifier is a candidate subsumer."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_more_general(self, first_classifier, second_classifier):
        """Determines if the first classifier is more general than the second
        classifier."""
        raise NotImplementedError


class XCSSubsumption(ISubsumptionStrategy):
    def __init__(self, rule_repr):
        self._rule_repr = rule_repr

    def does_subsume(self, subsumer, subsumee):
        """DOES SUBSUME function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        return subsumer.action == subsumee.action and \
            self._could_subsume(subsumer) and \
            self.is_more_general(subsumer, subsumee)

    def could_subsume(self, classifier):
        """COULD SUBSUME function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        return classifier.experience > get_hyperparam("theta_sub") and \
            classifier.error < get_hyperparam("epsilon_nought")

    def is_more_general(self, first_classifier, second_classifier):
        """IS MORE GENERAL function from
        'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002), modified to be rule representation
        agnostic."""
        first_generality = \
            self._rule_repr.calc_generality(first_classifier.condition)
        second_generality = \
            self._rule_repr.calc_generality(second_classifier.condition)
        first_contains_second = self._subsumer_contains_subsumee(
                subsumer=first_classifier_generality, subsumee=second_classifier)
        return (first_generality > second_generality) and \
                first_contains_second

    def _subsumer_contains_subsumee(self, subsumer, subsumee):
        """Second part of IS MORE GENERAL function from
        'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002), modified to be rule representation
        agnostic."""
        return self._rule_repr.check_condition_subsumption(
            subsumer.condition, subsumee.condition)

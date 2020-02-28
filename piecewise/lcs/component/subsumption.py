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
        return self.could_subsume(subsumer) and \
            subsumer.action == subsumee.action and \
            self.is_more_general(subsumer, subsumee) and \
            self.subsumer_contains_subsumee(subsumer,
                                            subsumee)

    def could_subsume(self, classifier):
        """COULD SUBSUME function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        return classifier.experience > get_hyperparam("theta_sub") and \
            classifier.error < get_hyperparam("epsilon_nought")

    def is_more_general(self, first_classifier, second_classifier):
        """First part of IS MORE GENERAL function from
        'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002), modified to be rule representation
        agnostic."""
        first_classifier_generality = \
            self._rule_repr.calc_generality(first_classifier.condition)
        second_classifier_generality = \
            self._rule_repr.calc_generality(second_classifier.condition)
        return first_classifier_generality > second_classifier_generality

    def subsumer_contains_subsumee(self, subsumer, subsumee):
        """Second part of IS MORE GENERAL function from
        'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002), modified to be rule representation
        agnostic."""
        return self._rule_repr.check_condition_subsumption(
            subsumer.condition, subsumee.condition)

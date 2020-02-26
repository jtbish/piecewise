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
    def is_more_general(self, general_classifier, specific_classifier):
        """Determines if the 'general' classifier is indeed more general than
        the 'specific' classifier."""
        raise NotImplementedError


class XCSSubsumption(ISubsumptionStrategy):
    def __init__(self, rule_repr):
        self._rule_repr = rule_repr

    def does_subsume(self, subsumer, subsumee):
        """DOES SUBSUME function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        return self.could_subsume(subsumer) and \
            self._have_same_actions(subsumer,
                                    subsumee) and \
            self.subsumer_is_more_general(subsumer,
                                          subsumee) and \
            self.subsumer_contains_subsumee(subsumer,
                                            subsumee)

    def could_subsume(self, classifier):
        """COULD SUBSUME function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        return classifier.experience > get_hyperparam("theta_sub") and \
            classifier.error < get_hyperparam("epsilon_nought")

    def _have_same_actions(self, subsumer, subsumee):
        return subsumer.action == subsumee.action

    def subsumer_is_more_general(self, subsumer, subsumee):
        """First part of IS MORE GENERAL function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002), modified to be rule representation
        agnostic."""
        subsumer_generality = \
            self._rule_repr.calc_generality(subsumer.condition)
        subsumee_generality = \
            self._rule_repr.calc_generality(subsumee.condition)
        return subsumer_generality > subsumee_generality

    def subsumer_contains_subsumee(self, subsumer, subsumee):
        """Second part of IS MORE GENERAL function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002), modified to be rule representation
        agnostic."""
        return self._rule_repr.first_contains_second(subsumer.condition,
                                                     subsumee.condition)

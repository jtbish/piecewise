import abc


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
    def __init__(self, rule_repr, hyperparams):
        self._rule_repr = rule_repr
        self._hyperparams = hyperparams

    def does_subsume(self, subsumer_classifier, subsumee_classifier):
        """DOES SUBSUME function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        return subsumer_classifier.action == subsumee_classifier.action and \
            self.could_subsume(subsumer_classifier) and \
            self.is_more_general(subsumer_classifier, subsumee_classifier)

    def could_subsume(self, classifier):
        """COULD SUBSUME function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        return classifier.experience > self._hyperparams["theta_sub"] and \
            classifier.error < self._hyperparams["epsilon_nought"]

    def is_more_general(self, general_classifier, specific_classifier):
        """IS MORE GENERAL function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002), modified to be rule representation
        agnostic."""
        if general_classifier.num_wildcards(self._rule_repr) <= \
                specific_classifier.num_wildcards(self._rule_repr):
            return False
        return self._is_actually_more_general(general_classifier,
                                              specific_classifier)

    def _is_actually_more_general(self, general_classifier,
                                  specific_classifier):
        for elem_idx, (general_condition_elem, specific_condition_elem) in \
            enumerate(zip(
                general_classifier.condition, specific_classifier.condition)):
            if not self._rule_repr.is_wildcard(
                    general_condition_elem, elem_idx) and \
                    general_condition_elem != \
                    specific_condition_elem:
                return False
        return True

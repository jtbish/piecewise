import functools

from piecewise.error.population_error import InvalidSizeError

from .abstract_classifier_set import AbstractClassifierSet, verify_membership
from .population_state import PopulationState


def delete_after(method):
    """Decorator to perform deletions after duplications or insertions into the
    population, in order to keep the size of the population at or below
    its upper bound."""
    @functools.wraps(method)
    def _delete_after(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self._perform_necessary_deletions()
        assert self._MIN_MICROS <= self.num_micros() <= self._max_micros
        return result

    return _delete_after


class Population(AbstractClassifierSet):
    def __init__(self, max_micros, deletion_strat):
        self._validate_max_micros(max_micros)
        self._max_micros = max_micros
        self._deletion_strat = deletion_strat
        self._state = PopulationState()
        super().__init__()

    def _validate_max_micros(self, max_micros):
        max_micros = int(max_micros)
        if not max_micros > 0:
            raise InvalidSizeError("Invalid max micros for population size: "
                                   f"{max_micros}, must be positive integer")

    @delete_after
    def insert(self, classifier, track_as=None):
        """INSERT IN POPULATION function from 'An Algorithmic Description of
        XCS' (Butz and Wilson, 2002)."""
        was_absorbed = self._try_to_absorb(classifier)
        if not was_absorbed:
            self._members.append(classifier)
            self._state.update(track_as, increment=classifier.numerosity)

    def _try_to_absorb(self, classifier):
        for member in self._members:
            if member.rule == classifier.rule:
                member.numerosity += classifier.numerosity
                self._state.update("absorption",
                                   increment=classifier.numerosity)
                return True
        return False

    @delete_after
    @verify_membership
    def duplicate(self, classifier, num_copies=1, track_as=None):
        classifier.numerosity += num_copies
        self._state.update(track_as, increment=num_copies)

    def _perform_necessary_deletions(self):
        """DELETE FROM POPULATION function from 'An Algorithmic Description of
        XCS' (Butz and Wilson, 2002).

        Delegates to the deletion strategy for actually selecting a classifier
        to 'delete', then does the 'deletion' via removing a single copy of the
        classifier from the population."""
        deletion_is_required = self.num_micros() >= self._max_micros
        if deletion_is_required:
            num_deletions = self.num_micros() - self._max_micros
            assert num_deletions >= 0
            for _ in range(num_deletions):
                classifier_to_delete = self._deletion_strat(self)
                self._remove_single_copy(classifier_to_delete)
                self._state.update("deletion", increment=1)

    @verify_membership
    def _remove_single_copy(self, classifier):
        """Lines 11-14 in body of second loop of
        DELETE FROM POPULATION function from 'An Algorithmic Description of
        XCS' (Butz and Wilson, 2002)."""
        if classifier.numerosity > 1:
            classifier.numerosity -= 1
        else:
            # can't have 0 numerosity, so remove member completely
            self._remove(classifier)

    @verify_membership
    def replace(self, replacee, replacer, track_as=None):
        self._remove(replacee)
        self.duplicate(replacer,
                       num_copies=replacee.numerosity,
                       track_as=track_as)

    def _remove(self, classifier):
        self._members.remove(classifier)

import functools

from piecewise.error.population_error import InvalidSizeError

from .abstract_classifier_set import AbstractClassifierSet, verify_membership
from .population_state import PopulationState


def delete_after(public_method):
    """Decorator to perform deletions after performing an operation on the
    population, in order to keep the size of the population at or below
    its upper bound."""
    @functools.wraps(public_method)
    def _delete_after(self, *args, **kwargs):
        result = public_method(self, *args, **kwargs)
        self._perform_necessary_deletions()
        assert self._MIN_MICROS <= self.num_micros() <= self._max_micros
        return result

    return _delete_after


def track_state_change(atomic_method):
    """Decorator to track state changes (in units of number of
    microclassifiers) when performing atomic operations on the population.
    """
    @functools.wraps(atomic_method)
    def _track_state_change(self, *args, track_label=None):
        micros_before = self.num_micros()
        result = atomic_method(self, *args, track_label)
        micros_after = self.num_micros()
        micros_delta = micros_after - micros_before
        self._state.update(track_label, micros_delta)
        return result

    return _track_state_change


class Population(AbstractClassifierSet):
    def __init__(self, max_micros, deletion_strat, rule_repr):
        self._validate_max_micros(max_micros)
        self._max_micros = max_micros
        self._deletion_strat = deletion_strat
        self._state = PopulationState()
        self._rule_repr = rule_repr
        super().__init__()

    def _validate_max_micros(self, max_micros):
        max_micros = int(max_micros)
        if not max_micros > 0:
            raise InvalidSizeError("Invalid max micros for population size: "
                                   f"{max_micros}, must be positive integer")

    @property
    def rule_repr(self):
        return self._rule_repr

    @delete_after
    def add(self, classifier, track_label=None):
        self._atomic_add_new(classifier, track_label=track_label)

    @delete_after
    def insert(self, classifier, track_label=None):
        """INSERT IN POPULATION function from 'An Algorithmic Description of
        XCS' (Butz and Wilson, 2002)."""
        was_absorbed = self._try_to_absorb(classifier)
        if not was_absorbed:
            self._atomic_add_new(classifier, track_label=track_label)

    def _try_to_absorb(self, classifier):
        for member in self._members:
            if member.rule == classifier.rule:
                self._absorb(classifier, member)
                return True
        return False

    def _absorb(self, absorbee, absorber):
        num_copies = absorbee.numerosity
        self._atomic_copy_existing(absorber,
                                   num_copies,
                                   track_label="absorption")

    @delete_after
    @verify_membership
    def duplicate(self, classifier, num_copies=1, track_label=None):
        self._atomic_copy_existing(classifier,
                                   num_copies,
                                   track_label=track_label)

    @verify_membership
    def replace(self, replacee, replacer, track_label=None):
        self._atomic_remove_whole(replacee)
        num_copies = replacee.numerosity
        self._atomic_copy_existing(replacer,
                                   num_copies,
                                   track_label=track_label)

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
                self._atomic_remove_single_copy(classifier_to_delete,
                                                track_label="deletion")

    # Atomic operations
    @track_state_change
    def _atomic_add_new(self, new_classifier, track_label):
        self._members.append(new_classifier)

    @track_state_change
    def _atomic_copy_existing(self, existing_classifier, num_copies,
                              track_label):
        assert num_copies >= 1
        for _ in range(num_copies):
            existing_classifier.numerosity += 1

    @track_state_change
    def _atomic_remove_whole(self, classifier, track_label):
        self._members.remove(classifier)

    @track_state_change
    def _atomic_remove_single_copy(self, existing_classifier, track_label):
        """Lines 11-14 in body of second loop of
        DELETE FROM POPULATION function from 'An Algorithmic Description of
        XCS' (Butz and Wilson, 2002)."""
        if existing_classifier.numerosity > 1:
            existing_classifier.numerosity -= 1
        else:
            # can't have 0 numerosity, so remove completely
            self._members.remove(existing_classifier)

    @property
    def state(self):
        return self._state

import functools

from piecewise.error.population_error import InvalidSizeError

from .abstract_classifier_set import AbstractClassifierSet, verify_membership
from .population_operation_recorder import PopulationOperationRecorder


def record_operation(atomic_method):
    """Decorator to record operations types (in units of number of
    microclassifiers) when performing atomic operations on the population.
    """
    @functools.wraps(atomic_method)
    def _record_operation(self, *args, operation_label=None):
        micros_before = self.num_micros
        result = atomic_method(self, *args, operation_label=operation_label)
        micros_after = self.num_micros
        micros_delta = micros_after - micros_before
        num_micros_altered = abs(micros_delta)

        if operation_label is not None:
            self._operation_recorder[operation_label] += num_micros_altered
        return result

    return _record_operation


class Population(AbstractClassifierSet):
    def __init__(self, max_micros):
        self._validate_max_micros(max_micros)
        self._max_micros = max_micros
        self._operation_recorder = \
            PopulationOperationRecorder()
        super().__init__()

    def _validate_max_micros(self, max_micros):
        max_micros = int(max_micros)
        if not max_micros > 0:
            raise InvalidSizeError("Invalid max micros for population size: "
                                   f"{max_micros}, must be positive integer")

    @property
    def max_micros(self):
        return self._max_micros

    @property
    def operations_record(self):
        return self._operation_recorder

    def add(self, classifier, *, operation_label=None):
        self._atomic_add_new(classifier, operation_label=operation_label)

    def insert(self, classifier, *, operation_label=None):
        """INSERT IN POPULATION function from 'An Algorithmic Description of
        XCS' (Butz and Wilson, 2002)."""
        was_absorbed = self._try_to_absorb(classifier)
        if not was_absorbed:
            self._atomic_add_new(classifier, operation_label=operation_label)

    def _try_to_absorb(self, classifier):
        for member in self._members:
            if member.rule == classifier.rule:
                self._absorb(classifier, member)
                return True
        return False

    def _absorb(self, absorbee, absorber):
        num_absorber_copies = absorbee.numerosity
        self._atomic_copy_existing(absorber,
                                   num_absorber_copies,
                                   operation_label="absorption")

    @verify_membership
    def duplicate(self, classifier, num_copies=1, *, operation_label=None):
        self._atomic_copy_existing(classifier,
                                   num_copies,
                                   operation_label=operation_label)

    @verify_membership
    def replace(self, replacee, replacer, *, operation_label=None):
        self._atomic_remove_whole(replacee)
        num_replacer_copies = replacee.numerosity
        self._atomic_copy_existing(replacer,
                                   num_replacer_copies,
                                   operation_label=operation_label)

    @verify_membership
    def delete(self, classifier):
        self._atomic_remove_single_copy(classifier, operation_label="deletion")

    # Atomic operations
    @record_operation
    def _atomic_add_new(self, new_classifier, *, operation_label=None):
        self._members.append(new_classifier)
        self._inc_num_micros(new_classifier.numerosity)

    @record_operation
    def _atomic_copy_existing(self,
                              existing_classifier,
                              num_copies,
                              *,
                              operation_label=None):
        assert num_copies >= 1
        existing_classifier.numerosity += num_copies
        self._inc_num_micros(num_copies)

    @record_operation
    def _atomic_remove_whole(self, classifier, *, operation_label=None):
        self._members.remove(classifier)
        self._dec_num_micros(classifier.numerosity)

    @record_operation
    def _atomic_remove_single_copy(self,
                                   existing_classifier,
                                   *,
                                   operation_label=None):
        """Lines 11-14 in body of second loop of
        DELETE FROM POPULATION function from 'An Algorithmic Description of
        XCS' (Butz and Wilson, 2002)."""
        if existing_classifier.numerosity > 1:
            existing_classifier.numerosity -= 1
        else:
            # can't have 0 numerosity, so remove completely
            self._members.remove(existing_classifier)
        self._dec_num_micros(1)

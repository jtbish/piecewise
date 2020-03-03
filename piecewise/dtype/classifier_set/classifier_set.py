from .classifier_set_base import ClassifierSetBase, verify_membership
import copy


class ClassifierSet(ClassifierSetBase):
    """Container used to store internal collections of classifiers, i.e.
    match set, action set, etc.

    This data structure acts as a "view" into the population, and so the
    references held in it are different from the population - changes in
    the population are not reflected in this set.

    This is desirable particularly for action sets in multi-step XCS where
    updating shared references between update steps can cause the state of
    the action set to be incongruent with its actual contents (num_micros
    as recorded might be incorrect).
    """
    def __init__(self):
        super().__init__()

    def add(self, classifier):
        """Adds the given classifier to the set.

        Take a copy of the classifier to destroy the reference to the
        population."""
        classifier = copy.deepcopy(classifier)
        self._members.append(classifier)
        self._inc_num_micros(classifier.numerosity)

    @verify_membership
    def remove(self, classifier):
        """Removes the given classifier from the set.

        Throws:
            MemberNotFoundError: if the classifier is not in the set.
        """
        self._members.remove(classifier)
        self._dec_num_micros(classifier.numerosity)

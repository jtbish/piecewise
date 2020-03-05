from .classifier_set_base import ClassifierSetBase, verify_membership


class ClassifierSet(ClassifierSetBase):
    """Container used to store internal collections of classifiers, i.e.
    match set, action set, etc.

    Notably the num_micros property for this class is calculated because
    this class acts as a "view" into the population - meaning that it stores
    refs to Classifier objs that are *most probably* also stored in the
    population somewhere.

    This means that these refs could be updated in some other scope without
    the knowledge of this set - thus keeping track of num_micros internally in
    this set is impossible/futile.
    """
    def __init__(self):
        super().__init__()

    @property
    def num_micros(self):
        return sum([classifier.numerosity for classifier in self._members])

    def add(self, classifier):
        """Adds the given classifier to the set."""
        self._members.append(classifier)

    @verify_membership
    def remove(self, classifier):
        """Removes the given classifier from the set.

        Throws:
            MemberNotFoundError: if the classifier is not in the set.
        """
        self._members.remove(classifier)

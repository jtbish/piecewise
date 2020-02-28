from .classifier_set_base import ClassifierSetBase, verify_membership


class ClassifierSet(ClassifierSetBase):
    """Container used to store internal collections of classifiers, i.e.
    match set, action set, etc.
    """
    def __init__(self):
        super().__init__()

    def add(self, classifier):
        """Adds the given classifier to the set."""
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

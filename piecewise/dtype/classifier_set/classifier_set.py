from .abstract_classifier_set import AbstractClassifierSet, verify_membership


class ClassifierSet(AbstractClassifierSet):
    """Concrete container used to store internal collections of classifiers, i.e.
    match set, action set, etc.
    """
    def __init__(self):
        super().__init__()

    def add(self, classifier):
        self._members.append(classifier)

    @verify_membership
    def remove(self, classifier):
        self._members.remove(classifier)

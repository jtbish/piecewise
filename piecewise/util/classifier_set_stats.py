"""Statistics and utility functions for classifier sets."""
_VALID_SUMMARY_STAT_TYPES = ("min", "mean", "max")


def calc_summary_stat(classifier_set, stat_type, classifier_property):
    """Calculates the given statistic type of the given classifier property."""
    assert stat_type in _VALID_SUMMARY_STAT_TYPES

    if stat_type == "min":
        return _calc_min(classifier_set, classifier_property)
    elif stat_type == "mean":
        return _calc_mean(classifier_set, classifier_property)
    elif stat_type == "max":
        return _calc_max(classifier_set, classifier_property)


def _calc_min(classifier_set, classifier_property):
    return min([
        getattr(classifier, classifier_property)
        for classifier in classifier_set
    ])


def _calc_mean(classifier_set, classifier_property):
    return sum([
        getattr(classifier, classifier_property) * classifier.numerosity
        for classifier in classifier_set
    ]) / classifier_set.num_micros


def _calc_max(classifier_set, classifier_property):
    return max([
        getattr(classifier, classifier_property)
        for classifier in classifier_set
    ])


def num_unique_actions(classifier_set):
    """Returns the cardinality of the unique actions set of the given
    classifier set.
    """
    return len(get_unique_actions_set(classifier_set))


def get_unique_actions_set(classifier_set):
    """Returns a set containing the unique actions advocated by the
    classifiers in the classifier set.
    """
    unique_actions = set()
    for classifier in classifier_set:
        unique_actions.add(classifier.rule.action)
    return unique_actions

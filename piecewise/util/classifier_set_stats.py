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
        getattr(classifier, classifier_property)
        for classifier in classifier_set
    ]) / classifier_set.num_micros()


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


def summarise_population(population, rule_repr, time_step):
    summary = {}
    summary["num_micros"] = population.num_micros()
    summary["num_macros"] = population.num_macros()
    summary["min_prediction"] = calc_summary_stat(population, "min",
                                                  "prediction")
    summary["mean_prediction"] = calc_summary_stat(population, "mean",
                                                   "prediction")
    summary["max_prediction"] = calc_summary_stat(population, "max",
                                                  "prediction")
    summary["min_error"] = calc_summary_stat(population, "min", "error")
    summary["mean_error"] = calc_summary_stat(population, "mean", "error")
    summary["max_error"] = calc_summary_stat(population, "max", "error")
    summary["min_fitness"] = calc_summary_stat(population, "min", "fitness")
    summary["mean_fitness"] = calc_summary_stat(population, "mean", "fitness")
    summary["max_fitness"] = calc_summary_stat(population, "max", "fitness")
    summary["min_time_stamp"] = calc_summary_stat(population, "min",
                                                  "time_stamp")
    summary["mean_time_stamp"] = calc_summary_stat(population, "mean",
                                                   "time_stamp")
    summary["max_time_stamp"] = calc_summary_stat(population, "max",
                                                  "time_stamp")
    summary["min_experience"] = calc_summary_stat(population, "min",
                                                  "experience")
    summary["mean_experience"] = calc_summary_stat(population, "mean",
                                                   "experience")
    summary["max_experience"] = calc_summary_stat(population, "max",
                                                  "experience")
    summary["min_action_set_size"] = \
        calc_summary_stat(population, "min", "action_set_size")
    summary["mean_action_set_size"] = \
        calc_summary_stat(population, "mean", "action_set_size")
    summary["max_action_set_size"] = \
        calc_summary_stat(population, "max", "action_set_size")
    summary["min_numerosity"] = \
        calc_summary_stat(population, "min", "numerosity")
    summary["max_numerosity"] = \
        calc_summary_stat(population, "max", "numerosity")
    summary["min_generality"] = min([
        classifier.generality_as_percentage(rule_repr)
        for classifier in population
    ])
    summary["mean_generality"] = sum([
        classifier.generality_as_percentage(rule_repr)
        for classifier in population
    ]) / population.num_micros()
    summary["max_generality"] = max([
        classifier.generality_as_percentage(rule_repr)
        for classifier in population
    ])
    summary["time_step"] = time_step
    return summary

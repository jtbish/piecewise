import abc
from collections import OrderedDict

from piecewise.util.classifier_set_stats import calc_summary_stat


class AbstractSubMonitor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, lcs):
        raise NotImplementedError

    # TODO remove
    @abc.abstractmethod
    def report(self):
        raise NotImplementedError


class TrainingPerformanceSubMonitor(AbstractSubMonitor):
    def __init__(self):
        self._training_performance_history = OrderedDict()

    def update(self, lcs):
        epoch_num = lcs.epoch_num
        training_performance = lcs.calc_training_performance()
        self._training_performance_history[epoch_num] = training_performance

    def report(self):
        last_key = next(reversed(self._training_performance_history))
        last_performance = self._training_performance_history[last_key]
        print(f"Training performance: {last_performance}")


class PopulationSummarySubMonitor(AbstractSubMonitor):
    def __init__(self):
        self._population_summaries = OrderedDict()

    def update(self, lcs):
        time_step = lcs.time_step
        summary = self._summarise_population(lcs)
        self._population_summaries[time_step] = summary

    def _summarise_population(self, lcs):
        population = lcs.population
        rule_repr = lcs.rule_repr
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
        summary["min_fitness"] = calc_summary_stat(population, "min",
                                                   "fitness")
        summary["mean_fitness"] = calc_summary_stat(population, "mean",
                                                    "fitness")
        summary["max_fitness"] = calc_summary_stat(population, "max",
                                                   "fitness")
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
        return summary

    def report(self):
        print("Population summary:")
        last_key = next(reversed(self._population_summaries))
        last_summary = self._population_summaries[last_key]
        for k, v in last_summary.items():
            if isinstance(v, float):
                print(f"{k}: {v:.2f}")
            else:
                print(f"{k}: {v}")


class PopulationOperationsSubMonitor(AbstractSubMonitor):
    def __init__(self):
        self._population_operations_history = OrderedDict()

    def update(self, lcs):
        time_step = lcs.time_step
        population = lcs.population
        operations_state = population.state.as_dict()
        self._population_operations_history[time_step] = operations_state

    def report(self):
        print("Population operations:")
        last_key = next(reversed(self._population_operations_history))
        print(self._population_operations_history[last_key])

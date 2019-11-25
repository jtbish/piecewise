import abc
from collections import OrderedDict

import matplotlib.pyplot as plt

from piecewise.util.classifier_set_stats import calc_summary_stat


class AbstractSubMonitor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, lcs):
        raise NotImplementedError

    # TODO remove
    @abc.abstractmethod
    def report(self):
        raise NotImplementedError

    @abc.abstractmethod
    def plot(self):
        pass


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

    def plot(self):
        plt.figure()
        epoch_nums = list(self._training_performance_history.keys())
        accuracy_vals = list(self._training_performance_history.values())
        plt.plot(epoch_nums, accuracy_vals)
        plt.title("Training performance")
        plt.xlabel("Epoch num")
        plt.ylabel("Training set accuracy")
        plt.savefig("training_performance.png")


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

    def plot(self):
        # what is plottable on the same graph?
        # mean error/1000, mean prediction/1000
        # macros/max_micros, micros/max_micros
        # generality
        plt.figure()
        time_steps = list(self._population_summaries.keys())
        mean_error_vals = [
            summary["mean_error"] / 1000
            for summary in self._population_summaries.values()
        ]
        mean_prediction_vals = [
            summary["mean_prediction"] / 1000
            for summary in self._population_summaries.values()
        ]
        micro_vals = [
            summary["num_micros"] / 400
            for summary in self._population_summaries.values()
        ]
        macro_vals = [
            summary["num_macros"] / 400
            for summary in self._population_summaries.values()
        ]
        generality_vals = [
            summary["mean_generality"] / 100
            for summary in self._population_summaries.values()
        ]
        fitness_vals = [
            summary["mean_fitness"]
            for summary in self._population_summaries.values()
        ]
        plt.plot(time_steps, mean_error_vals, label="mean error")
        plt.plot(time_steps, mean_prediction_vals, label="mean prediction")
        plt.plot(time_steps, micro_vals, label="micros")
        plt.plot(time_steps, macro_vals, label="macros")
        plt.plot(time_steps, generality_vals, label="generality")
        plt.plot(time_steps, fitness_vals, label="fitness")
        plt.xlabel("Time step")
        plt.ylabel("Normalised value")
        plt.title("Population summary trends")
        plt.legend(loc="upper right")
        plt.savefig("pop_summary.png")


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

    def plot(self):
        plt.figure()
        time_steps = list(self._population_operations_history.keys())
        last_key = next(reversed(self._population_operations_history))
        recorded_operations = \
            tuple(self._population_operations_history[last_key].keys())
        for operation in recorded_operations:
            operation_vals = []
            for time_step in time_steps:
                val = self._population_operations_history[time_step].get(
                    operation, 0)
                operation_vals.append(abs(val))
            plt.plot(time_steps, operation_vals, label=operation)
        plt.xlabel("Time step")
        plt.ylabel("Cumulative number of operations")
        plt.yscale("log")
        plt.title("Population operations")
        plt.legend(loc="upper right")
        plt.savefig("pop_ops.png")

import abc
from collections import OrderedDict, namedtuple

import matplotlib.pyplot as plt

from piecewise.util.classifier_set_stats import calc_summary_stat


class AbstractSubMonitor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, lcs, epoch_num):
        raise NotImplementedError

    @abc.abstractmethod
    def query(self):
        raise NotImplementedError

    @abc.abstractmethod
    def report(self):
        raise NotImplementedError


class TrainingPerformanceSubMonitor(AbstractSubMonitor):
    def __init__(self):
        self._training_performance_history = OrderedDict()

    def update(self, lcs, epoch_num):
        training_performance = lcs.calc_training_performance()
        self._training_performance_history[epoch_num] = training_performance

    def query(self):
        return self._training_performance_history

    def report(self):
        plt.figure()
        epoch_nums = list(self._training_performance_history.keys())
        training_performance_results = \
            list(self._training_performance_history.values())
        plt.plot(epoch_nums, training_performance_results)
        plt.xlabel("Epoch number")
        plt.ylabel("Training set performance")
        plt.title("Training set performance vs. time")
        plt.savefig("training_performance.png")


class PopulationSizeSubMonitor(AbstractSubMonitor):
    def __init__(self):
        self._population_size_history = OrderedDict()

    def update(self, lcs, epoch_num):
        num_micros = lcs.population.num_micros()
        num_macros = lcs.population.num_macros()
        self._population_size_history[epoch_num] = {
            "num micros": num_micros,
            "num macros": num_macros
        }

    def query(self):
        return self._population_size_history

    def report(self):
        plt.figure()
        epoch_nums = list(self._population_size_history.keys())
        micros_values = [
            history["num micros"]
            for history in self._population_size_history.values()
        ]
        macros_values = [
            history["num macros"]
            for history in self._population_size_history.values()
        ]
        plt.plot(epoch_nums, micros_values, label="num micros")
        plt.plot(epoch_nums, macros_values, label="num macros")
        plt.xlabel("Epoch number")
        plt.ylabel("Population size")
        plt.title("Population size vs. time")
        plt.legend()
        plt.savefig("pop_size.png")


ClassifierSetStat = namedtuple("ClassifierSetStat",
                               ["stat_type", "classifier_property"])


class PopulationStatisticsSubMonitor(AbstractSubMonitor):
    def __init__(self, stats_to_monitor):
        self._stats_to_monitor = stats_to_monitor
        self._stat_summary_history = OrderedDict()

    def update(self, lcs, epoch_num):
        summary = self._gen_summary(lcs.population)
        self._stat_summary_history[epoch_num] = summary

    def _gen_summary(self, population):
        summary = {}
        for stat_to_monitor in self._stats_to_monitor:
            result = self._calc_stat(population, stat_to_monitor)
            stat_name = self._get_stat_name(stat_to_monitor)
            summary[stat_name] = result
        return summary

    def _calc_stat(self, population, stat):
        return calc_summary_stat(population, stat.stat_type,
                                 stat.classifier_property)

    def _get_stat_name(self, stat):
        return stat.stat_type + " " + stat.classifier_property

    def query(self):
        return self._stat_summary_history

    def report(self):
        pass


class PopulationOperationsSubMonitor(AbstractSubMonitor):
    def __init__(self):
        self._population_operations_history = OrderedDict()

    def update(self, lcs, epoch_num):
        population = lcs.population
        operations_state = population.get_operations_state()
        self._population_operations_history[epoch_num] = operations_state

    def query(self):
        return self._population_operations_history

    def report(self):
        pass

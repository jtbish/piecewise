import abc
from collections import namedtuple

from piecewise.util.classifier_set_stats import calc_summary_stat

EpochData = namedtuple("EpochData", ["epoch_num", "recorded_data"])


class AbstractSubMonitor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, lcs, epoch_num):
        raise NotImplementedError

    @abc.abstractmethod
    def query(self):
        raise NotImplementedError


class TrainingPerformanceSubMonitor(AbstractSubMonitor):
    def __init__(self):
        self._training_performance_history = []

    def update(self, lcs, epoch_num):
        training_performance = lcs.calc_training_performance()
        self._training_performance_history.append(
            EpochData(epoch_num, training_performance))

    def query(self):
        return self._training_performance_history


class PopulationSizeSubMonitor(AbstractSubMonitor):
    def __init__(self):
        self._population_size_history = []

    def update(self, lcs, epoch_num):
        num_micros = lcs.population.num_micros()
        num_macros = lcs.population.num_macros()
        self._population_size_history.append(
            EpochData(epoch_num, {
                "num_micros": num_micros,
                "num_macros": num_macros
            }))

    def query(self):
        return self._population_size_history


ClassifierSetStat = namedtuple("ClassifierSetStat",
                               ["stat_type", "classifier_property"])


class PopulationStatisticsSubMonitor(AbstractSubMonitor):
    def __init__(self, stats_to_monitor):
        self._stats_to_monitor = stats_to_monitor
        self._stat_summary_history = []

    def update(self, lcs, epoch_num):
        summary = self._gen_summary(lcs.population)
        self._stat_summary_history.append(EpochData(epoch_num, summary))

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


class PopulationOperationsSubMonitor(AbstractSubMonitor):
    def __init__(self):
        self._population_operations_history = []

    def update(self, lcs, epoch_num):
        population = lcs.population
        pop_ops = population.operations()
        self._population_operations_history.append(
            EpochData(epoch_num, pop_ops))

    def query(self):
        return self._population_operations_history

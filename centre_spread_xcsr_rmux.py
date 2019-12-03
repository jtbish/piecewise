import math

import matplotlib.pyplot as plt
import numpy as np

from piecewise.algorithm import make_xcs
from piecewise.codec import NullCodec
from piecewise.environment import RealMultiplexer
from piecewise.experiment import Experiment
from piecewise.lcs import SupervisedLCS
from piecewise.monitor import Monitor, MonitorItem
from piecewise.rule_repr import CentreSpreadRuleRepr
from piecewise.util.classifier_set_stats import calc_summary_stat

MONITOR_ITEMS = [
    MonitorItem("num_micros", lambda lcs: lcs.population.num_micros),
    MonitorItem("num_macros", lambda lcs: lcs.population.num_macros),
    MonitorItem("training_acc", lambda lcs: lcs.calc_training_performance()),
    MonitorItem(
        "mean_error",
        lambda lcs: calc_summary_stat(lcs.population, "mean", "error")),
    MonitorItem(
        "max_fitness",
        lambda lcs: calc_summary_stat(lcs.population, "max", "fitness")),
    MonitorItem("deletion",
                lambda lcs: lcs.population.operations_record["deletion"]),
    MonitorItem("covering",
                lambda lcs: lcs.population.operations_record["covering"]),
    MonitorItem(
        "as_subsumption",
        lambda lcs: lcs.population.operations_record["as_subsumption"]),
    MonitorItem(
        "ga_subsumption",
        lambda lcs: lcs.population.operations_record["ga_subsumption"]),
    MonitorItem("discovery",
                lambda lcs: lcs.population.operations_record["discovery"]),
    MonitorItem("absorption",
                lambda lcs: lcs.population.operations_record["absorption"])
]

NUM_MUX_ADDRESS_BITS = 2
NUM_MUX_SAMPLES = 64


def main():
    seeds = list(range(10))
    experiments = [_make_experiment(seed) for seed in seeds]
    monitor_outputs = []
    for experiment in experiments:
        population, monitor_output = experiment.run()
        monitor_outputs.append(monitor_output)
    _plot_monitor_outputs(monitor_outputs)


def _make_experiment(seed):
    lcs = _make_lcs()
    training_instances_per_epoch = 64
    desired_total_training_instances = 20000
    num_epochs = math.ceil(desired_total_training_instances /
                           training_instances_per_epoch)
    monitor = Monitor(*MONITOR_ITEMS)
    return Experiment(lcs, seed, num_epochs, monitor)


def _make_lcs():
    balanced_thresholds = [0.5] * 6
    env = RealMultiplexer(num_address_bits=NUM_MUX_ADDRESS_BITS,
                          shuffle_dataset=True,
                          num_samples=NUM_MUX_SAMPLES,
                          seed=0,
                          thresholds=balanced_thresholds)
    codec = NullCodec()
    situation_space = codec.make_situation_space(env.obs_space)
    rule_repr = CentreSpreadRuleRepr(situation_space)
    hyperparams = {
        "N": 800,
        "beta": 0.2,
        "alpha": 0.1,
        "epsilon_nought": 0.01,
        "nu": 5,
        "gamma": 0.71,
        "theta_ga": 12,
        "chi": 0.8,
        "mu": 0.04,
        "theta_del": 20,
        "delta": 0.1,
        "theta_sub": 20,
        "p_wildcard": 0.33,
        "prediction_I": 1e-3,
        "epsilon_I": 1e-3,
        "fitness_I": 1e-3,
        "p_explore": 0.5,
        "theta_mna": len(env.action_set),
        "do_ga_subsumption": True,
        "do_as_subsumption": True,
        "s_nought": 1.0,
        "m": 0.1
    }
    algorithm = make_xcs(env.step_type, env.action_set, rule_repr, hyperparams)
    return SupervisedLCS(env, codec, algorithm)


def _plot_monitor_outputs(monitor_outputs):
    _plot_population_operations(monitor_outputs)
    _plot_remainder(monitor_outputs)


def _plot_population_operations(monitor_outputs):
    plt.figure()
    for monitor_item_name in ("covering", "absorption", "discovery",
                              "deletion", "as_subsumption", "ga_subsumption"):
        aggregated_experiment_data = \
            _aggregate_experiments_for_monitor_item(monitor_item_name,
                                                    monitor_outputs)
        mean_value_over_experiments = \
            np.mean(aggregated_experiment_data, axis=0)
        plt.plot(mean_value_over_experiments, label=monitor_item_name)
    plt.ylabel("Cumulative number of operations")
    plt.yscale("log")
    plt.xlabel("Epoch num")
    plt.title("Population operations during training")
    plt.legend()
    plt.savefig("pop_ops.png")


def _plot_remainder(monitor_outputs):
    plt.figure()
    _plot_pop_sizes(monitor_outputs)
    _plot_stats(monitor_outputs)
    _plot_training_accuracy(monitor_outputs)
    plt.xlabel("Epoch num")
    plt.title("Learning diagnostics")
    plt.legend()
    plt.savefig("learning_diagnostics.png")


def _plot_pop_sizes(monitor_outputs):
    mean_micros = \
        np.mean(_aggregate_experiments_for_monitor_item("num_micros",
                monitor_outputs), axis=0)
    mean_macros = \
        np.mean(_aggregate_experiments_for_monitor_item("num_macros",
                monitor_outputs), axis=0)
    plt.plot(mean_micros / 800, label="num micros / pop size")
    plt.plot(mean_macros / 800, label="num macros / pop size")


def _plot_stats(monitor_outputs):
    mean_of_mean_error = \
        np.mean(_aggregate_experiments_for_monitor_item("mean_error",
                monitor_outputs), axis=0)
    mean_of_max_fitness = \
        np.mean(_aggregate_experiments_for_monitor_item("max_fitness",
                monitor_outputs), axis=0)
    plt.plot(mean_of_mean_error / 1000, label="mean error / max error")
    plt.plot(mean_of_max_fitness, label="max fitness")


def _plot_training_accuracy(monitor_outputs):
    mean_training_accuracy = \
        np.mean(_aggregate_experiments_for_monitor_item("training_acc",
                monitor_outputs), axis=0)
    plt.plot(mean_training_accuracy / 100, label="training accuracy")


def _aggregate_experiments_for_monitor_item(monitor_item_name,
                                            monitor_outputs):
    aggregated_experiment_data = []
    for monitor_output in monitor_outputs:
        aggregated_experiment_data.append(monitor_output[monitor_item_name])
    return aggregated_experiment_data


if __name__ == "__main__":
    main()

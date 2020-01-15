import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import piecewise
from piecewise.algorithm import make_xcs
from piecewise.codec import NullCodec
from piecewise.environment import RealMultiplexer
from piecewise.experiment import Experiment
from piecewise.lcs import ClassificationLCS
from piecewise.monitor import Monitor, MonitorItem
from piecewise.rule_repr import CentreSpreadRuleRepr
from piecewise.util.classifier_set_stats import calc_summary_stat

#alg_seeds = list(range(10))
alg_seeds = [1]

total_training_instances_per_experiment = 20000

# 6-rmux
rmux_params = {
    "num_address_bits": 2,
    "shuffle_dataset": True,
    "num_samples": 500,  # large data sample
    "seed": 0,
    "thresholds": [0.5] * 6,  # balanced thresholds
    "reward_correct": 1000,
    "reward_incorrect": 0
}

xcsr_hyperparams = {
    "N": 800,
    "beta": 0.2,
    "alpha": 0.1,
    "epsilon_nought": 10,
    "nu": 5,
    "gamma": 0.71,
    "theta_ga": 12,
    "chi": 0.8,
    "mu": 0.04,
    "theta_del": 20,
    "delta": 0.1,
    "theta_sub": 20,
    "p_wildcard": 0.33,
    "prediction_I": 0.01,
    "epsilon_I": 1,
    "fitness_I": 0.01,
    "p_explore": 0.5,
    "theta_mna": 2,  # binary classification
    "do_ga_subsumption": True,
    "do_as_subsumption": True,
    "m": 0.1,
    "s_nought": 1.0
}

monitor_items = [
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

piecewise.dtype.constants.FLOAT_ALLELE_EQ_REL_TOL = 0.01


def main():
    experiments = [_make_experiment(alg_seed) for alg_seed in alg_seeds]
    monitor_outputs = []
    for idx, experiment in enumerate(experiments):
        print(f"Runnning experiment {idx+1}")
        _, monitor_output = experiment.run()
        monitor_outputs.append(monitor_output)
        experiment.archive()
        print(f"Finished experiment {idx+1}")
    _plot_monitor_outputs(monitor_outputs)


def _make_experiment(alg_seed):
    lcs = _make_lcs(alg_seed)
    training_instances_per_epoch = rmux_params["num_samples"]
    num_epochs = math.ceil(total_training_instances_per_experiment /
                           training_instances_per_epoch)
    monitor = Monitor(*monitor_items)
    tag = f"alg_seed_{alg_seed}"
    return Experiment(tag, lcs, num_epochs, monitor)


def _make_lcs(alg_seed):
    env = RealMultiplexer(**rmux_params)
    codec = NullCodec()
    rule_repr = CentreSpreadRuleRepr(codec.make_situation_space(env.obs_space))
    alg = make_xcs(env.step_type, env.action_set, rule_repr, xcsr_hyperparams)
    alg.set_seed(alg_seed)
    return ClassificationLCS(env, codec, alg)


def _plot_monitor_outputs(monitor_outputs):
    _make_plot_dir()
    _plot_population_operations(monitor_outputs)
    _plot_learning_diagnostics(monitor_outputs)


def _make_plot_dir():
    plot_dir = Path("./plot")
    plot_dir.mkdir(exist_ok=True)


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
    plt.savefig("plot/xcsr_rmux_pop_ops.png")


def _plot_learning_diagnostics(monitor_outputs):
    plt.figure()
    _plot_pop_sizes(monitor_outputs)
    _plot_stats(monitor_outputs)
    _plot_training_accuracy(monitor_outputs)
    plt.xlabel("Epoch num")
    plt.title("Learning diagnostics during training")
    plt.legend()
    plt.savefig("plot/xcsr_rmux_learn_diag.png")


def _plot_pop_sizes(monitor_outputs):
    mean_micros = \
        np.mean(_aggregate_experiments_for_monitor_item("num_micros",
                monitor_outputs), axis=0)
    mean_macros = \
        np.mean(_aggregate_experiments_for_monitor_item("num_macros",
                monitor_outputs), axis=0)
    plt.plot(mean_micros / xcsr_hyperparams["N"],
             label="num micros / pop size")
    plt.plot(mean_macros / xcsr_hyperparams["N"],
             label="num macros / pop size")


def _plot_stats(monitor_outputs):
    mean_of_mean_error = \
        np.mean(_aggregate_experiments_for_monitor_item("mean_error",
                monitor_outputs), axis=0)
    mean_of_max_fitness = \
        np.mean(_aggregate_experiments_for_monitor_item("max_fitness",
                monitor_outputs), axis=0)
    max_error = rmux_params["reward_correct"]
    plt.plot(mean_of_mean_error / max_error, label="mean error / max error")
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

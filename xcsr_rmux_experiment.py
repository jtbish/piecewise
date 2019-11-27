import math
import random

import matplotlib.pyplot as plt
import numpy as np

from piecewise.algorithm import make_xcs
from piecewise.codec import NullCodec
from piecewise.environment import RealMultiplexer
from piecewise.experiment import Experiment
from piecewise.lcs import SupervisedLCS
from piecewise.monitor import (ClassifierSetStat, Monitor,
                               PopulationOperationsSubMonitor,
                               PopulationSizeSubMonitor,
                               PopulationStatisticsSubMonitor,
                               TrainingPerformanceSubMonitor)
from piecewise.rule_repr import CentreSpreadRuleRepr


def main():
    monitor_results = []
    alg_seeds = list(range(10))
    rmux_seed = 0
    num_rmux_samples = 64
    for alg_seed in alg_seeds:
        random.seed(alg_seed)
        np.random.seed(alg_seed)
        # 6-rmux
        env = RealMultiplexer(num_address_bits=2,
                              shuffle_dataset=True,
                              num_samples=num_rmux_samples,
                              seed=rmux_seed,
                              thresholds=[0.5] * 6)
        codec = NullCodec()
        situation_space = codec.make_situation_space(env.obs_space)
        rule_repr = CentreSpreadRuleRepr(situation_space)
        num_actions = len(env.action_set)
        xcs_hyperparams = {
            "N": 800,
            "beta": 0.2,
            "alpha": 0.1,
            "epsilon_nought": 0.01,
            "nu": 5,
            "gamma": 0.71,
            "theta_ga": 25,
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
            "theta_mna": num_actions,
            "do_ga_subsumption": True,
            "do_as_subsumption": True,
            "s_nought": 1.0,
            "m": 0.1
        }
        algorithm = make_xcs(env.step_type, env.action_set, rule_repr,
                             xcs_hyperparams)
        monitor = Monitor(
            TrainingPerformanceSubMonitor(), PopulationOperationsSubMonitor(),
            PopulationSizeSubMonitor(),
            PopulationStatisticsSubMonitor([
                ClassifierSetStat("mean", "error"),
                ClassifierSetStat("max", "fitness")
            ]))
        lcs = SupervisedLCS(env, codec, algorithm)

        num_epochs = math.ceil(20000 / num_rmux_samples)  # 20k probs on 6-rmux
        experiment = Experiment(lcs, monitor, num_epochs)
        population, monitor_result = experiment.run()
        print_top_sixteen_fitness(population)
        monitor_results.append(monitor_result)

    merged_results = merge_monitor_results(monitor_results)
    plot(merged_results)


def print_top_sixteen_fitness(population):
    fitness_sorted = sorted(population,
                            key=lambda classifier: classifier.fitness,
                            reverse=True)
    for i in range(16):
        print(fitness_sorted[i])
    print("\n")


def merge_monitor_results(monitor_results):
    monitor_merged = {}
    for sub_monitor in ("TrainingPerformanceSubMonitor",
                        "PopulationOperationsSubMonitor",
                        "PopulationSizeSubMonitor",
                        "PopulationStatisticsSubMonitor"):
        sub_monitor_raw = []
        for monitor_result in monitor_results:
            sub_monitor_result = monitor_result[sub_monitor]
            sub_monitor_raw.append(sub_monitor_result)
        sub_monitor_merged = merge_dicts(sub_monitor_raw)
        monitor_merged[sub_monitor] = sub_monitor_merged
    return monitor_merged


def merge_dicts(dicts):
    merged = {}
    keys = list(dicts[0].keys())
    for key in keys:
        vals = []
        for dict_ in dicts:
            vals.append(dict_[key])
        merged[key] = vals
    return merged


def plot(merged_results):
    for sub_monitor in ("TrainingPerformanceSubMonitor",
                        "PopulationOperationsSubMonitor",
                        "PopulationSizeSubMonitor",
                        "PopulationStatisticsSubMonitor"):
        if sub_monitor == "TrainingPerformanceSubMonitor":
            plot_training_performance(merged_results[sub_monitor])
        elif sub_monitor == "PopulationOperationsSubMonitor":
            plot_pop_ops(merged_results[sub_monitor])
        elif sub_monitor == "PopulationSizeSubMonitor":
            plot_pop_size(merged_results[sub_monitor])
        else:
            plot_pop_stats(merged_results[sub_monitor])


def plot_training_performance(merged_performances):
    plt.figure()
    epoch_nums = list(merged_performances.keys())
    mean_performance_vals = np.mean(list(merged_performances.values()), axis=1)
    plt.plot(epoch_nums, mean_performance_vals)
    plt.xlabel("Epoch num")
    plt.ylabel("Training accuracy")
    plt.title("Training accuracy over time")
    plt.savefig("training_accuracy.png")


def plot_pop_ops(merged_pop_ops):
    plt.figure()
    epoch_nums = list(merged_pop_ops.keys())
    for pop_op in ("covering", "absorption", "discovery", "ga_subsumption",
                   "action_set_subsumption", "deletion"):
        aggregate_all_epoch = []
        for epoch_num in epoch_nums:
            aggregate_single_epoch = []
            for dict_ in merged_pop_ops[epoch_num]:
                pop_op_val = dict_.get(pop_op, 0)
                aggregate_single_epoch.append(pop_op_val)
            aggregate_all_epoch.append(np.mean(aggregate_single_epoch))
        plt.plot(epoch_nums, aggregate_all_epoch, label=pop_op)
    plt.xlabel("Epoch num")
    plt.ylabel("Cumulative number of operations")
    plt.yscale("log")
    plt.title("Population operations over time")
    plt.legend()
    plt.savefig("pop_ops.png")


def plot_pop_size(merged_pop_sizes):
    plt.figure()
    epoch_nums = list(merged_pop_sizes.keys())
    for sub_key in ("num micros", "num macros"):
        aggregate_all_epoch = []
        for epoch_num in epoch_nums:
            aggregate_single_epoch = []
            for dict_ in merged_pop_sizes[epoch_num]:
                size_val = dict_[sub_key]
                aggregate_single_epoch.append(size_val)
            aggregate_all_epoch.append(np.mean(aggregate_single_epoch))
        plt.plot(epoch_nums, aggregate_all_epoch, label=sub_key)
    plt.xlabel("Epoch num")
    plt.ylabel("Count")
    plt.title("Pop size over time")
    plt.legend()
    plt.savefig("pop_size.png")


def plot_pop_stats(merged_pop_stats):
    plt.figure()
    epoch_nums = list(merged_pop_stats.keys())
    for sub_key in ("mean error", "max fitness"):
        aggregate_all_epoch = []
        for epoch_num in epoch_nums:
            aggregate_single_epoch = []
            for dict_ in merged_pop_stats[epoch_num]:
                stat_val = dict_[sub_key]
                aggregate_single_epoch.append(stat_val)
            if sub_key == "mean error":
                # normalise
                aggregate_all_epoch.append(
                    np.mean(aggregate_single_epoch) / 1000)
            else:
                aggregate_all_epoch.append(np.mean(aggregate_single_epoch))
        plt.plot(epoch_nums, aggregate_all_epoch, label=sub_key)
    plt.xlabel("Epoch num")
    plt.title("Pop stats over time")
    plt.legend()
    plt.savefig("pop_stats.png")


if __name__ == "__main__":
    main()

#!/usr/bin/python3
import logging

from piecewise.algorithm import make_canonical_xcs
from piecewise.environment import make_real_mux_env
from piecewise.experiment import Experiment
from piecewise.monitor import Monitor, MonitorItem
from piecewise.rule_repr import make_centre_spread_rule_repr
from piecewise.util.classifier_set_stats import calc_summary_stat


def main():
    # 6-mux
    env = make_real_mux_env(thresholds=[0.5] * 6,
                            num_address_bits=2,
                            shuffle_dataset=True,
                            shuffle_seed=0,
                            num_samples=100,
                            data_gen_seed=0,
                            reward_correct=1000,
                            reward_incorrect=0)

    rule_repr = make_centre_spread_rule_repr(env)

    alg_hyperparams = {
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
        "prediction_I": 1e-3,
        "epsilon_I": 1e-3,
        "fitness_I": 1e-3,
        "p_explore": 0.5,
        "theta_mna": len(env.action_set),
        "do_ga_subsumption": True,
        "do_as_subsumption": True,
        "m": 0.1,
        "s_nought": 1.0
    }
    alg = make_canonical_xcs(env, rule_repr, alg_hyperparams, seed=0)

    monitor_items = [
        MonitorItem("num_micros",
                    lambda experiment: experiment.population.num_micros),
        MonitorItem("num_macros",
                    lambda experiment: experiment.population.num_macros),
        MonitorItem(
            "performance",
            lambda experiment: experiment.calc_performance(strat="accuracy")),
        MonitorItem(
            "mean_error", lambda experiment: calc_summary_stat(
                experiment.population, "mean", "error")),
        MonitorItem(
            "max_fitness", lambda experiment: calc_summary_stat(
                experiment.population, "max", "fitness")),
        MonitorItem(
            "deletion", lambda experiment: experiment.population.
            operations_record["deletion"]),
        MonitorItem(
            "covering", lambda experiment: experiment.population.
            operations_record["covering"]),
        MonitorItem(
            "as_subsumption", lambda experiment: experiment.population.
            operations_record["as_subsumption"]),
        MonitorItem(
            "ga_subsumption", lambda experiment: experiment.population.
            operations_record["ga_subsumption"]),
        MonitorItem(
            "discovery", lambda experiment: experiment.population.
            operations_record["discovery"]),
        MonitorItem(
            "absorption", lambda experiment: experiment.population.
            operations_record["absorption"])
    ]
    monitor = Monitor(monitor_items, update_freq=100)

    experiment = Experiment(save_dir="rmux",
                            env=env,
                            alg=alg,
                            num_training_samples=1000,
                            monitor=monitor,
                            logging_level=logging.DEBUG)
    experiment.run()
    experiment.save()


if __name__ == "__main__":
    main()

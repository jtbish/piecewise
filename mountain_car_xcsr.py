#!/usr/bin/python3
import argparse
import logging

from piecewise.environment import make_continuous_mountain_car_env
from piecewise.experiment import Experiment
from piecewise.lcs import make_custom_xcs_from_canonical_base
from piecewise.lcs.component import DecayingEpsilonGreedy
from piecewise.rule_repr import make_continuous_min_percentage_rule_repr

#from piecewise.dtype import config as dtype_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    return parser.parse_args()


def main(args):
    env = make_continuous_mountain_car_env(normalise=True, seed=0)

    rule_repr = make_continuous_min_percentage_rule_repr(env)

    lcs_hyperparams = {
        "N": 1000,
        "beta": 0.1,
        "alpha": 0.1,
        "epsilon_nought": 0.025,
        "nu": 5,
        "gamma": 0.95,
        "theta_ga": 25,
        "chi": 0.8,
        "mu": 0.25,
        "theta_del": 20,
        "delta": 0.1,
        "theta_sub": 5,
        "prediction_I": 1e-3,
        "epsilon_I": 1e-3,
        "fitness_I": 1e-3,
        "e_greedy_decay_factor": 0.9995,
        "theta_mna": len(env.action_set),
        "do_ga_subsumption": True,
        "do_as_subsumption": True,
        "m": 0.05,  # mutation magnitude
        "s_nought": 0.25  # covering spread magnitude
    }
    lcs = make_custom_xcs_from_canonical_base(
        env,
        rule_repr,
        lcs_hyperparams,
        seed=0,
        action_selection=DecayingEpsilonGreedy())

    experiment = Experiment(name=args.experiment_name,
                            env=env,
                            lcs=lcs,
                            num_training_samples=1000,
                            use_lcs_monitor=True,
                            lcs_monitor_freq=100,
                            use_loop_monitor=True,
                            logging_level=logging.DEBUG,
                            var_args=args)
    experiment.run()
    experiment.save_results()


if __name__ == "__main__":
    args = parse_args()
    main(args)

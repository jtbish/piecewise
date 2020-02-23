#!/usr/bin/python3
import argparse
import logging

from piecewise.environment import make_cartpole_environment
from piecewise.experiment import Experiment
from piecewise.lcs import make_custom_xcs_from_canonical_base
from piecewise.lcs.component import DecayingEpsilonGreedy
from piecewise.rule_repr import make_centre_spread_rule_repr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    return parser.parse_args()


def main(args):
    env = make_cartpole_environment(seed=0)

    rule_repr = make_centre_spread_rule_repr(env)

    lcs_hyperparams = {
        "N": 1000,
        "beta": 0.1,
        "alpha": 0.1,
        "epsilon_nought": 0.01,
        "nu": 5,
        "gamma": 0.95,
        "theta_ga": 25,
        "chi": 0.8,
        "mu": 0.4,
        "theta_del": 20,
        "delta": 0.1,
        "theta_sub": 20,
        "prediction_I": 1e-3,
        "epsilon_I": 1e-3,
        "fitness_I": 1e-3,
        "e_greedy_decay_factor": 0.9999,
        "theta_mna": len(env.action_set),
        "do_ga_subsumption": True,
        "do_as_subsumption": True,
        "m": 0.1,
        "s_nought": 1.0
    }
    # TODO
    # m and s_nought are sensitive to the range of each feature.......
    lcs = make_custom_xcs_from_canonical_base(
        env,
        rule_repr,
        lcs_hyperparams,
        seed=0,
        action_selection=DecayingEpsilonGreedy())

    experiment = Experiment(name=args.experiment_name,
                            env=env,
                            lcs=lcs,
                            num_training_samples=40000,
                            use_lcs_monitor=True,
                            lcs_monitor_freq=1000,
                            use_loop_monitor=True,
                            logging_level=logging.DEBUG,
                            var_args=args)
    experiment.run()
    experiment.save_results()


if __name__ == "__main__":
    args = parse_args()
    main(args)

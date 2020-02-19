#!/usr/bin/python3
import argparse
import logging

from piecewise.algorithm import make_canonical_xcs
from piecewise.environment import make_discrete_mux_env
from piecewise.experiment import Experiment
from piecewise.rule_repr import DiscreteRuleRepr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-training-samples", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--epsilon-nought", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--prediction-I", type=float, required=True)
    parser.add_argument("--epsilon-I", type=float, required=True)
    parser.add_argument("--fitness-I", type=float, required=True)
    parser.add_argument("--alg-seed", type=int, required=True)
    parser.add_argument("--experiment-name", required=True)
    return parser.parse_args()


def main(args):
    # 6-mux
    env = make_discrete_mux_env(num_address_bits=2,
                                shuffle_dataset=True,
                                shuffle_seed=0,
                                reward_correct=1000,
                                reward_incorrect=0)

    rule_repr = DiscreteRuleRepr()

    alg_hyperparams = {
        "N": args.N,
        "beta": 0.2,
        "alpha": 0.1,
        "epsilon_nought": args.epsilon_nought,
        "nu": 5,
        "gamma": args.gamma,
        "theta_ga": 25,
        "chi": 0.8,
        "mu": 0.04,
        "theta_del": 20,
        "delta": 0.1,
        "theta_sub": 20,
        "p_wildcard": 0.33,
        "prediction_I": args.prediction_I,
        "epsilon_I": args.epsilon_I,
        "fitness_I": args.fitness_I,
        "p_explore": 0.5,
        "theta_mna": len(env.action_set),
        "do_ga_subsumption": True,
        "do_as_subsumption": True,
    }
    alg = make_canonical_xcs(env,
                             rule_repr,
                             alg_hyperparams,
                             seed=args.alg_seed)

    experiment = Experiment(name=args.experiment_name,
                            env=env,
                            alg=alg,
                            num_training_samples=args.num_training_samples,
                            use_population_monitor=True,
                            population_monitor_freq=100,
                            use_loop_monitor=True,
                            logging_level=logging.DEBUG,
                            var_args=args)
    experiment.run()
    experiment.save()


if __name__ == "__main__":
    args = parse_args()
    main(args)

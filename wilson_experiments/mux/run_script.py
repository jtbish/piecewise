#!/usr/bin/python3
import logging

from piecewise.algorithm import make_canonical_xcs
from piecewise.environment import make_discrete_mux_env
from piecewise.experiment import Experiment
from piecewise.rule_repr import DiscreteRuleRepr


def main():
    # 6-mux
    env = make_discrete_mux_env(num_address_bits=2,
                                shuffle_dataset=True,
                                shuffle_seed=0,
                                reward_correct=1000,
                                reward_incorrect=0)

    rule_repr = DiscreteRuleRepr()

    alg_hyperparams = {
        "N": 400,
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
        "theta_mna": len(env.action_set),
        "do_ga_subsumption": True,
        "do_as_subsumption": True,
    }
    alg = make_canonical_xcs(env, rule_repr, alg_hyperparams, seed=0)

    experiment = Experiment(save_dir="mux",
                            env=env,
                            alg=alg,
                            num_training_samples=1000,
                            use_population_monitor=True,
                            population_monitor_freq=100,
                            use_loop_monitor=True,
                            logging_level=logging.DEBUG)
    experiment.run()
    experiment.save()


if __name__ == "__main__":
    main()

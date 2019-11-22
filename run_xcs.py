from piecewise.algorithm import XCS
from piecewise.codec import NullCodec
from piecewise.environment import DiscreteMultiplexer
from piecewise.lcs import LCS
from piecewise.monitor import (Monitor, PopulationOperationsSubMonitor,
                               PopulationSummarySubMonitor,
                               TrainingPerformanceSubMonitor)
from piecewise.rule_repr import TernaryRuleRepr


def main():
    # 6-multiplexer
    env = DiscreteMultiplexer(num_address_bits=2, shuffle_dataset=True)
    codec = NullCodec()
    rule_repr = TernaryRuleRepr()
    num_actions = len(env.action_set)
    xcs_hyperparams = {
        "N": 400,
        "beta": 0.2,
        "alpha": 0.1,
        "epsilon_nought": 0.01,
        "nu": 5,
        "gamma": 0.71,
        "theta_ga": 25,
        "chi": 0.8,
        "mu": 0.4,
        "theta_del": 20,
        "delta": 0.1,
        "theta_sub": 20,
        "p_wildcard": 0.33,
        "prediction_I": 1e-3,
        "epsilon_I": 1e-3,
        "fitness_I": 1e-3,
        "theta_mna": num_actions,
        "do_ga_subsumption": True,
        "do_as_subsumption": True
    }
    algorithm = XCS(env.action_set, env.step_type, rule_repr, xcs_hyperparams)
    monitor = Monitor.from_sub_monitor_classes(TrainingPerformanceSubMonitor,
                                               PopulationSummarySubMonitor,
                                               PopulationOperationsSubMonitor)
    lcs = LCS(env, codec, rule_repr, algorithm, monitor)
    lcs.train(num_epochs=150)


if __name__ == "__main__":
    main()

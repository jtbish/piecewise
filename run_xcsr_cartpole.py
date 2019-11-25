from piecewise.algorithm import make_xcs
from piecewise.codec import NullCodec
from piecewise.environment import GymEnvironment
from piecewise.lcs import LCS
from piecewise.monitor import (Monitor, PopulationOperationsSubMonitor,
                               PopulationSummarySubMonitor,
                               TrainingPerformanceSubMonitor)
from piecewise.rule_repr import CentreSpreadRuleRepr


def main():
    # 6-multiplexer
    env = GymEnvironment("CartPole-v0", 26)
    print(env.obs_space)
    print(env.action_set)
    codec = NullCodec()
    situation_space = codec.make_situation_space(env.obs_space)
    rule_repr = CentreSpreadRuleRepr(situation_space)
    num_actions = len(env.action_set)
    xcs_hyperparams = {
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
        "theta_mna": num_actions,
        "do_ga_subsumption": True,
        "do_as_subsumption": True,
        "s_nought": 1.0,
        "m": 0.1
    }
    algorithm = make_xcs(env.step_type, env.action_set, rule_repr,
                         xcs_hyperparams)
    monitor = Monitor.from_sub_monitor_classes(PopulationSummarySubMonitor,
                                               PopulationOperationsSubMonitor)
    lcs = LCS(env, codec, rule_repr, algorithm, monitor)
    lcs.train(num_epochs=100)


if __name__ == "__main__":
    main()

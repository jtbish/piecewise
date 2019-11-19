from piecewise.algorithm import XCS
from piecewise.codec import NullCodec
from piecewise.environment import RealMultiplexer
from piecewise.lcs import LCS
from piecewise.rule_repr import CentreSpreadRuleRepr


def main():
    # 6-multiplexer
    thresholds = [0.5] * 6
    env = RealMultiplexer(num_address_bits=2,
                          shuffle_dataset=True,
                          num_samples=64,
                          thresholds=thresholds)
    codec = NullCodec()
    situation_space = codec.make_situation_space(env.obs_space)
    rule_repr = CentreSpreadRuleRepr(situation_space)
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
        "do_as_subsumption": True,
        "s_nought": 1.0,
        "m": 0.1
    }
    algorithm = XCS(env.action_set, rule_repr, xcs_hyperparams)
    lcs = LCS(env, codec, rule_repr, algorithm)
    lcs.train(num_epochs=100, monitor=True)


if __name__ == "__main__":
    main()

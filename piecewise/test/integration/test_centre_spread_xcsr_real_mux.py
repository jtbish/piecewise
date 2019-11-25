from piecewise.algorithm import make_xcs
from piecewise.codec import NullCodec
from piecewise.dtype import Population
from piecewise.environment import RealMultiplexer
from piecewise.environment.supervised.multiplexer.abstract_multiplexer import \
    calc_total_bits
from piecewise.lcs import LCS
from piecewise.rule_repr import CentreSpreadRuleRepr


class TestCentreSpreadXCSROnRealMultiplexer:
    def _setup_lcs(self):
        num_address_bits = 2
        total_bits = calc_total_bits(num_address_bits)
        balanced_thresholds = [0.5] * total_bits
        env = RealMultiplexer(num_address_bits=2,
                              num_samples=64,
                              shuffle_dataset=True,
                              thresholds=balanced_thresholds)
        codec = NullCodec()
        situation_space = codec.make_situation_space(env.obs_space)
        rule_repr = CentreSpreadRuleRepr(situation_space)
        algorithm = self._init_algorithm(env, rule_repr)
        self._lcs = LCS(env, codec, rule_repr, algorithm)

    def _init_algorithm(self, env, rule_repr):
        num_actions = len(env.action_set)
        hyperparams = {
            "N": 100,
            "beta": 0.1,
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
            "prediction_I": 1e-6,
            "epsilon_I": 1e-6,
            "fitness_I": 1e-6,
            "p_explore": 0.5,
            "theta_mna": num_actions,
            "do_ga_subsumption": True,
            "do_as_subsumption": True,
            "s_nought": 1.0,
            "m": 0.1
        }
        xcs = make_xcs(env.step_type, env.action_set, rule_repr, hyperparams)
        return xcs

    def test_training(self):
        self._setup_lcs()
        population = self._lcs.train(num_epochs=5)
        assert isinstance(population, Population)

from piecewise.algorithm import make_xcs
from piecewise.codec import NullCodec
from piecewise.environment import DiscreteMultiplexer
from piecewise.lcs import LCS
from piecewise.rule_repr import TernaryRuleRepr
from piecewise.dtype import Population


class TestTernaryXCSOnDiscreteMultiplexer:
    def _setup_lcs(self):
        env = DiscreteMultiplexer(num_address_bits=2, shuffle_dataset=True)
        codec = NullCodec()
        rule_repr = TernaryRuleRepr()
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
            "do_as_subsumption": True
        }
        xcs = make_xcs(env.step_type, env.action_set, rule_repr, hyperparams)
        return xcs

    def test_training(self):
        self._setup_lcs()
        population = self._lcs.train(num_epochs=5)
        assert isinstance(population, Population)

from piecewise import Experiment
from piecewise.algorithm import make_canonical_xcs
from piecewise.environment import make_discrete_mux_env
from piecewise.rule_repr import DiscreteRuleRepr
from piecewise.util import ParamRegistry

# 6-mux
env = make_discrete_mux_env(num_address_bits=2,
                            shuffle_dataset=True,
                            reward_correct=1000,
                            reward_incorrect=0)

rule_repr = DiscreteRuleRepr()

alg_hps = ParamRegistry()
alg_hps.register("N", 400, used_by=("xcs_root"))
alg_hps.register("beta", 0.2, used_by=("fitness_update", "credit_assignment"))
alg_hps.register("alpha", 0.1, used_by=("fitness_update"))
alg_hps.register("epsilon_nought",
                 0.01,
                 used_by=("fitness_update", "subsumption"))
alg_hps.register("nu", 5, used_by=("fitness_update"))
alg_hps.register("gamma", 0.71, used_by=("credit_assignment"))
alg_hps.register("theta_ga", 12, used_by=("xcs_root"))
alg_hps.register("chi", 0.8, used_by=("rule_discovery"))
alg_hps.register("mu", 0.04, used_by=("rule_discovery"))
alg_hps.register("theta_del", 20, used_by=("deletion"))
alg_hps.register("delta", 0.1, used_by=("deletion"))
alg_hps.register("theta_sub", 20, used_by=("subsumption"))
alg_hps.register("p_wildcard", 0.33, used_by=("covering"))
alg_hps.register("prediction_I", 1e-3, used_by=("covering"))
alg_hps.register("epsilon_I", 1e-3, used_by=("covering"))
alg_hps.register("fitness_I", 1e-3, used_by=("covering"))
alg_hps.register("p_explore", 0.5, used_by=("action_selection"))
alg_hps.register("theta_mna", len(env.action_set), used_by=("xcs_root"))
alg_hps.register("do_ga_subsumption", True, used_by=("rule_discovery"))
alg_hps.register("do_as_subsumption", True, used_by=("xcs_root"))

alg = make_canonical_xcs(env, rule_repr, alg_hps)

experiment_config = {
    "num_training_samples": 10000,
    "logging": "verbose"
}
experiment = Experiment(env, alg, **experiment_config)
experiment.run()
experiment.archive()

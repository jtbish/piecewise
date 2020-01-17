from piecewise import Experiment
from piecewise.algorithm import make_xcs
from piecewise.component import (EpsilonGreedy, FitnessWeightedAvgPrediction,
                                 RuleReprCovering, RuleReprMatching,
                                 XCSAccuracyFitnessUpdate, XCSCreditAssignment,
                                 XCSGeneticAlgorithm, XCSRouletteWheelDeletion,
                                 XCSSubsumption)
from piecewise.environment import make_discrete_mux_env
from piecewise.rule_repr import DiscreteRuleRepr
from piecewise.util import Hyperparams

env = make_discrete_mux_env(num_address_bits=2,
                            shuffle_dataset=True,
                            reward_correct=1000,
                            reward_incorrect=0)

rule_repr = DiscreteRuleRepr()

# register hyperparams
hps = Hyperparams()
hps.register("N", 400, used_by=("xcs_root"))
hps.register("beta", 0.2, used_by=("fitness_update", "credit_assignment"))
hps.register("alpha", 0.1, used_by=("fitness_update"))
hps.register("epsilon_nought", 0.01, used_by=("fitness_update", "subsumption"))
hps.register("nu", 5, used_by=("fitness_update"))
hps.register("gamma", 0.71, used_by=("credit_assignment"))
hps.register("theta_ga", 12, used_by=("xcs_root"))
hps.register("chi", 0.8, used_by=("rule_discovery"))
hps.register("mu", 0.04, used_by=("rule_discovery"))
hps.register("theta_del", 20, used_by=("deletion"))
hps.register("delta", 0.1, used_by=("deletion"))
hps.register("theta_sub", 20, used_by=("subsumption"))
hps.register("p_wildcard", 0.33, used_by=("covering"))
hps.register("prediction_I", 1e-3, used_by=("covering"))
hps.register("epsilon_I", 1e-3, used_by=("covering"))
hps.register("fitness_I", 1e-3, used_by=("covering"))
hps.register("p_explore", 0.5, used_by=("action_selection"))
hps.register("theta_mna", len(env.action_set), used_by=("xcs_root"))
hps.register("do_ga_subsumption", True, used_by=("rule_discovery"))
hps.register("do_as_subsumption", True, used_by=("xcs_root"))
hps.register("m", 0.1, used_by=("rule_discovery"))
hps.register("s_nought", 1.0, used_by=("covering"))

# make alg from components
matching = RuleReprMatching(rule_repr)
covering = RuleReprCovering(env.action_set, rule_repr)
prediction = FitnessWeightedAvgPrediction(env.action_set)
action_selection = EpsilonGreedy(hps.get("action_selection"))
credit_assignment = XCSCreditAssignment(hps.get("credit_assignment"))
fitness_update = XCSAccuracyFitnessUpdate(hps.get("fitness_update"))
subsumption = XCSSubsumption(rule_repr, hps.get("subsumption"))
rule_discovery = XCSGeneticAlgorithm(env.action_set, rule_repr, subsumption,
                                     hps.get("rule_discovery"))
deletion = XCSRouletteWheelDeletion(hps.get("deletion"))

alg = make_xcs(env.step_type, matching, covering, prediction, action_selection,
               credit_assignment, fitness_update, subsumption, rule_discovery,
               deletion, hps.get("xcs_root"))

experiment = Experiment()
experiment.run()
experiment.archive()

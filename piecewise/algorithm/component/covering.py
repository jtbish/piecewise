from piecewise.algorithm.hyperparams import hyperparams_registry as hps_reg
from piecewise.algorithm.rng import np_random
from piecewise.dtype import Classifier, Rule
from piecewise.util.classifier_set_stats import get_unique_actions_set


class RuleReprCovering:
    """Provides an implementation of rule representation dependent covering.

    Generating the covering condition involves knowledge of wildcards, and so
    is delegated to the rule representation.
    """
    def __init__(self, env_action_set, rule_repr):
        self._env_action_set = env_action_set
        self._rule_repr = rule_repr

    def __call__(self, match_set, situation, time_step):
        """Remaining part of GENERATE COVERING CLASSIFIER function from
        'An Algorithmic Description of XCS' (Butz and Wilson, 2002).

        First part (generation of the covering condition) is delegated to the
        rule representation."""
        covering_condition = self._rule_repr.gen_covering_condition(situation)
        covering_action = self._gen_covering_action(match_set)

        rule = Rule(covering_condition, covering_action)
        classifier = Classifier(rule, hps_reg["prediction_I"],
                                hps_reg["epsilon_I"], hps_reg["fitness_I"],
                                time_step)
        return classifier

    def _gen_covering_action(self, match_set):
        possible_covering_actions = \
            tuple(self._env_action_set - get_unique_actions_set(match_set))
        assert len(possible_covering_actions) > 0
        return np_random.choice(possible_covering_actions)

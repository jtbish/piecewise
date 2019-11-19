import abc
import random

from piecewise.dtype import Classifier, Rule
from piecewise.util.classifier_set_stats import get_unique_actions_set


class CoveringStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, env_action_set=None, hyperparams=None):
        self._env_action_set = env_action_set
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def __call__(self, match_set, situation, time_step):
        """Return a single covering classifier which matches the given
        situation."""
        raise NotImplementedError


class RuleReprCovering(CoveringStrategy):
    """Provides an implementation of rule representation dependent covering.

    Generating the covering condition involves knowledge of wildcards, and so
    is delegated to the rule representation.
    """
    def __init__(self, env_action_set, rule_repr, hyperparams):
        self._rule_repr = rule_repr
        super().__init__(env_action_set, hyperparams)

    def __call__(self, match_set, situation, time_step):
        """Remaining part of GENERATE COVERING CLASSIFIER function from
        'An Algorithmic Description of XCS' (Butz and Wilson, 2002).

        First part (generation of the covering condition) is delegated to the
        rule representation."""
        covering_condition = self._rule_repr.gen_covering_condition(
            situation, self._hyperparams)
        possible_covering_actions = \
            self._env_action_set - get_unique_actions_set(match_set)
        covering_action = random.choice(tuple(possible_covering_actions))

        rule = Rule(covering_condition, covering_action)
        classifier = Classifier(rule, self._hyperparams["prediction_I"],
                                self._hyperparams["epsilon_I"],
                                self._hyperparams["fitness_I"], time_step)
        return classifier

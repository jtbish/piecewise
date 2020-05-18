import abc

from piecewise.dtype import Classifier, LinearPredictionClassifier, Rule
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng
from piecewise.util.classifier_set_stats import (get_unique_actions_set,
                                                 num_unique_actions)


# Factories for specific classifier types
def make_classifier(rule, time_step):
    return Classifier(rule, get_hyperparam("prediction_I"),
                      get_hyperparam("epsilon_I"), get_hyperparam("fitness_I"),
                      time_step)


def make_linear_prediction_classifier(rule, time_step):
    return LinearPredictionClassifier(rule, get_hyperparam("epsilon_I"),
                                      get_hyperparam("fitness_I"), time_step,
                                      get_hyperparam("x_nought"), get_rng())


class RuleReprCoveringABC(metaclass=abc.ABCMeta):
    def __init__(self, env_action_set, rule_repr, classifier_factory):
        self._env_action_set = env_action_set
        self._rule_repr = rule_repr
        self._classifier_factory = classifier_factory

    @abc.abstractmethod
    def __call__(self, population, match_set, situation, time_step):
        raise NotImplementedError


class RuleReprCovering(RuleReprCoveringABC):
    """Provides an implementation of rule representation dependent covering.

    Generating the covering condition is delegated to the rule
    representation."""
    def __call__(self, population, match_set, situation, time_step):
        """Second loop of GENERATE MATCH SET function from
        'An Algorithmic Description of XCS' (Butz and Wilson, 2002)."""
        while self._should_cover(match_set):
            covering_classifier = self._gen_covering_classifier(
                match_set, situation, time_step)
            population.add(covering_classifier, operation_label="covering")
            match_set.add(covering_classifier)

    def _should_cover(self, match_set):
        return num_unique_actions(match_set) < get_hyperparam("theta_mna")

    def _gen_covering_classifier(self, match_set, situation, time_step):
        """Remaining part of GENERATE COVERING CLASSIFIER function from
        'An Algorithmic Description of XCS' (Butz and Wilson, 2002).

        First part (generation of the covering condition) is delegated to the
        rule representation."""
        covering_condition = self._rule_repr.gen_covering_condition(situation)
        covering_action = self._gen_covering_action(match_set)
        rule = Rule(covering_condition,
                    covering_action,
                    num_features=len(situation))
        classifier = self._classifier_factory(rule, time_step)
        return classifier

    def _gen_covering_action(self, match_set):
        possible_covering_actions = \
            tuple(self._env_action_set - get_unique_actions_set(match_set))
        assert len(possible_covering_actions) > 0
        return get_rng().choice(possible_covering_actions)

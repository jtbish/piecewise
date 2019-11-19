import abc
import random


class MutationStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, hyperparams=None):
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def __call__(self, classifier, situation):
        """Performs mutation on the specified classifier (in-place), using
        information from the situation.
        """
        raise NotImplementedError


class RuleReprMutation(MutationStrategy):
    """Provides an implementation of rule representation dependent mutation.

    Mutating condition elements involves knowledge of wildcards, and so
    is delegated to the rule representation.
    """
    def __init__(self, rule_repr, env_action_set, hyperparams):
        self._rule_repr = rule_repr
        self._env_action_set = env_action_set
        super().__init__(hyperparams)

    def __call__(self, classifier, situation):
        """Second part (action mutation) of APPLY MUTATION function from 'An
        Algorithmic Description of XCS' (Butz and Wilson, 2002).

        First part (condition mutation) is delegated to the rule
        representation."""
        self._rule_repr.mutate_condition(classifier.condition,
                                         self._hyperparams, situation)
        self._mutate_action(classifier)

    def _mutate_action(self, classifier):
        should_mutate_action = random.random() < self._hyperparams["mu"]
        if should_mutate_action:
            possible_actions = self._env_action_set - {classifier.action}
            classifier.action = random.choice(tuple(possible_actions))

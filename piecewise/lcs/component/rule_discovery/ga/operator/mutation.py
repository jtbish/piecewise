class RuleReprMutation:
    """Provides an implementation of rule representation dependent mutation.

    Mutating condition elements involves knowledge of wildcards, and so
    is delegated to the rule representation.
    """
    def __init__(self, env_action_set, rule_repr):
        self._env_action_set = env_action_set
        self._rule_repr = rule_repr

    def __call__(self, classifier, situation):
        """Second part (action mutation) of APPLY MUTATION function from 'An
        Algorithmic Description of XCS' (Butz and Wilson, 2002).

        First part (condition mutation) is delegated to the rule
        representation."""
        self._rule_repr.mutate_condition(classifier.condition, situation)

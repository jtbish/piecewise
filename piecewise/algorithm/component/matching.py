from piecewise.dtype import ClassifierSet


class RuleReprMatching:
    """Provides an implementation of rule representation dependent matching.

    Such matching may involve taking into account the wildcards of the rule
    representation or the structure of the condition elements, and so the
    responsibility of determining a match is delegated to the specific rule
    representation being used.
    """
    def __init__(self, rule_repr):
        self._rule_repr = rule_repr

    def __call__(self, population, situation):
        """First loop of GENERATE MATCH SET function from
        'An Algorithmic Description of XCS' (Butz and Wilson, 2002)."""
        match_set = ClassifierSet()
        for classifier in population:
            if self._rule_repr.does_match(classifier.condition, situation):
                match_set.add(classifier)
        return match_set

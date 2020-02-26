class Condition:
    """Represents the 'IF' (antecedent) part of a classifier rule."""
    def __init__(self, genotype):
        self._genotype = genotype

    @property
    def genotype(self):
        return self._genotype

    def phenotype(self, rule_repr):
        return rule_repr.map_genotype_to_phenotype(self._genotype)

    def __eq__(self, other):
        return self._genotype == other._genotype

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"{self._genotype!r})"

    def __str__(self):
        return f"{self._genotype}"

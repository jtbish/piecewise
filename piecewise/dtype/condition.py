class Condition:
    """Mutable sequence type that represents a sequence of alleles."""
    def __init__(self, rule_repr, alleles):
        self._rule_repr = rule_repr
        self._alleles = list(alleles)

    @classmethod
    def from_allele_args(cls, *alleles):
        return cls(alleles)

    def __eq__(self, other):
        for (my_allele, other_allele) in zip(self._alleles, other._alleles):
            if my_allele != other_allele:
                return False
        return True

    def __setitem__(self, idx, value):
        self._alleles[idx] = value

    def __getitem__(self, idx):
        return self._alleles[idx]

    def __len__(self):
        return len(self._alleles)

    def __iter__(self):
        return iter(self._alleles)

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"{self._alleles!r})"

    def __str__(self):
        return self._rule_repr.genotype_to_phenotype(self)

import numpy as np

from .config import float_allele_rel_tol


class Genotype:
    """Mutable sequence type that represents a sequence of alleles."""
    def __init__(self, alleles):
        self._alleles = list(alleles)

    @classmethod
    def from_allele_args(cls, *alleles):
        return cls(alleles)

    def count(self, allele_value):
        return self._alleles.count(allele_value)

    def __eq__(self, other):
        for (my_allele, other_allele) in zip(self._alleles, other._alleles):
            if not self._alleles_are_equal(my_allele, other_allele):
                return False
        return True

    def _alleles_are_equal(self, my_allele, other_allele):
        # TODO subclass?
        are_floats = isinstance(my_allele, np.floating) and \
                isinstance(other_allele, np.floating)
        if are_floats:
            return np.isclose(my_allele,
                              other_allele,
                              rtol=float_allele_rel_tol)
        else:
            return my_allele == other_allele

    def __setitem__(self, idx, value):
        self._alleles[idx] = value

    def __getitem__(self, key):
        if isinstance(key, slice):
            # slicing, return tuple of contents
            start, stop, step = key.indices(len(self))
            return tuple([self[i] for i in range(start, stop, step)])
        else:
            key = int(key)
            # normal indexing
            return self._alleles[key]

    def __len__(self):
        return len(self._alleles)

    def __iter__(self):
        return iter(self._alleles)

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"{self._alleles!r})"

    def __str__(self):
        return "(" + ", ".join([str(allele) for allele in self._alleles]) + ")"

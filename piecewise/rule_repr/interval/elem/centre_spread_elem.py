import random

from .interval_elem import IntervalElem


class CentreSpreadElem(IntervalElem):
    def __init__(self, centre_allele, spread_allele):
        self._centre_allele = centre_allele
        self._spread_allele = spread_allele

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{self._centre_allele!r}, "
                f"{self._spread_allele!r})")

    def __str__(self):
        return f"({self._centre_allele}, {self._spread_allele})"

    def __eq__(self, other):
        return self._centre_allele == other._centre_allele and \
                self._spread_allele == other._spread_allele

    def lower(self):
        return self._centre_allele - self._spread_allele

    def upper(self):
        return self._centre_allele + self._spread_allele

    def mutate(self, hyperparams):
        for allele in (self._centre_allele, self._spread_allele):
            self._mutate_allele(allele, hyperparams)

    def _mutate_allele(self, allele, hyperparams):
        """Implementation of mutation for XCSR as described in 'Get Real! XCS
        With Continuous-Valued Inputs' (Wilson, 2000)."""
        should_mutate = random.random() < hyperparams["mu"]
        if should_mutate:
            adjustment_magnitude = random.uniform(0, hyperparams["m"])
            adjustment_sign = random.choice([1, -1])
            adjustment_amount = adjustment_magnitude * adjustment_sign
            allele += adjustment_amount

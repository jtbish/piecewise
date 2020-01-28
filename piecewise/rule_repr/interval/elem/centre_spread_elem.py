from piecewise.algorithm.rng import np_random
from piecewise.algorithm.hyperparams import hyperparams_registry as hps_reg

from .interval_elem import IntervalElemABC


class CentreSpreadElem(IntervalElemABC):
    """Represents a (centre, spread) tuple."""
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

    def mutate(self):
        for allele in (self._centre_allele, self._spread_allele):
            self._mutate_allele(allele)

    def _mutate_allele(self, allele):
        """Implementation of mutation for XCSR as described in 'Get Real! XCS
        With Continuous-Valued Inputs' (Wilson, 2000)."""
        should_mutate = np_random.rand() < hps_reg["mu"]
        if should_mutate:
            adjustment_magnitude = np_random.uniform(0, hps_reg["m"])
            adjustment_sign = np_random.choice([1, -1])
            adjustment_amount = adjustment_magnitude * adjustment_sign
            allele += adjustment_amount

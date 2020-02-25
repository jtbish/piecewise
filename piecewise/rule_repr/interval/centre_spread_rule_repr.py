from piecewise.dtype import Condition, FloatAllele
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng

from .interval_rule_repr import IntervalRuleReprABC
from .predicate.centre_spread_predicate import CentreSpreadPredicate


def make_centre_spread_rule_repr(env):
    return CentreSpreadRuleRepr(situation_space=env.obs_space,
                                env_dtype=env.dtype)


class CentreSpreadRuleRepr(IntervalRuleReprABC):
    """Rule representation that works with (centre, spread) interval
    predicates."""
    def __init__(self, situation_space, env_dtype):
        super().__init__(situation_space, env_dtype)

    def _create_wildcard_predicate_for_dim(self, dimension):
        centre = (dimension.upper + dimension.lower) / 2
        spread = dimension.upper - centre
        return self._make_predicate(centre, spread)

    def _make_predicate(self, centre, spread):
        return CentreSpreadPredicate(centre, spread)

    def gen_covering_condition(self, situation):
        alleles = []
        for situation_elem in situation:
            centre = situation_elem
            spread = get_rng().uniform(0, get_hyperparam("s_nought"))
            alleles.append(self._allele_type(centre))
            alleles.append(self._allele_type(spread))
        return Condition(rule_repr=self, alleles=alleles)

    def mutate_condition(self, condition, situation=None):
        for allele in condition:
            self._mutate_allele(allele)

    def _mutate_allele(self, allele):
        should_mutate = get_rng().rand() < get_hyperparam("mu")
        if should_mutate:
            adjustment_magnitude = get_rng().uniform(0, get_hyperparam("m"))
            adjustment_sign = get_rng().choice([1, -1])
            adjustment_amount = adjustment_magnitude * adjustment_sign
            allele += adjustment_amount

    def genotype_to_phenotype(self, condition):
        assert len(condition) % 2 == 0
        phenotype = []
        for centre_idx in range(0, len(condition), 2):
            spread_idx = centre_idx + 1
            centre = condition[centre_idx].value
            spread = condition[spread_idx].value
            phenotype.append(self._make_predicate(centre, spread))
        return tuple(phenotype)

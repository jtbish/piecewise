from piecewise.dtype import Condition, FloatAllele, Genotype
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng

from .interval import Interval
from .interval_rule_repr import IntervalRuleReprABC


def make_centre_spread_rule_repr(env):
    return CentreSpreadRuleRepr(situation_space=env.obs_space)


class CentreSpreadRuleRepr(IntervalRuleReprABC):
    """Rule representation that works with (centre, spread) interval
    predicates."""
    def __init__(self, situation_space):
        super().__init__(situation_space)

    def _gen_covering_condition(self, situation):
        alleles = []
        for situation_elem in situation:
            centre = situation_elem
            spread = get_rng().uniform(0, get_hyperparam("s_nought"))
            alleles.append(FloatAllele(centre))
            alleles.append(FloatAllele(spread))
        genotype = Genotype(alleles)
        return Condition(genotype)

    def mutate_condition(self, condition, situation=None):
        for allele in condition.genotype:
            self._mutate_allele(allele)

    def _mutate_allele(self, allele):
        should_mutate = get_rng().rand() < get_hyperparam("mu")
        if should_mutate:
            adjustment_magnitude = get_rng().uniform(0, get_hyperparam("m"))
            adjustment_sign = get_rng().choice([1, -1])
            adjustment_amount = adjustment_magnitude * adjustment_sign
            allele += adjustment_amount

    def _make_phenotype_interval_from_allele_vals(self,
                                                  first_val,
                                                  second_val,
                                                  situation_space_idx=None):
        centre = first_val
        spread = second_val
        lower = centre - spread
        upper = centre + spread
        return Interval(lower, upper)

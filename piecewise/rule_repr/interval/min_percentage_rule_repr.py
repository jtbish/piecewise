from piecewise.dtype import Condition, FloatAllele, Genotype
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng

from .interval import Interval
from .interval_rule_repr import IntervalRuleReprABC


def make_min_percentage_rule_repr(env):
    return MinPercentageRuleRepr(situation_space=env.obs_space)


class MinPercentageRuleRepr(IntervalRuleReprABC):
    """Rule representation that works with (min, percent_to_max) interval
    predicates."""
    def __init__(self, situation_space):
        super().__init__(situation_space)

    def _gen_covering_condition(self, situation):
        alleles = []
        for (idx, situation_elem) in enumerate(situation):
            lower = situation_elem - get_rng().uniform(
                0, get_hyperparam("s_nought"))
            upper = situation_elem + get_rng().uniform(
                0, get_hyperparam("s_nought"))
            assert lower <= upper

            dimension = self._situation_space[idx]
            frac_to_upper = self._calc_frac_to_upper(lower, upper,
                                                     dimension.upper)

            alleles.append(FloatAllele(lower))
            alleles.append(FloatAllele(frac_to_upper))
        genotype = Genotype(alleles)
        self._enforce_valid_genotype(genotype)
        return Condition(genotype)

    def _calc_frac_to_upper(self, lower, upper, dimension_upper):
        frac_to_upper = (upper - lower) / (dimension_upper - lower)
        assert 0.0 <= frac_to_upper <= 1.0
        return frac_to_upper

    def _calc_upper(self, lower, frac_to_upper, dimension_upper):
        span_from_lower = (dimension_upper - lower) * frac_to_upper
        upper = lower + span_from_lower
        assert upper <= dimension_upper
        return upper

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        super().crossover_conditions(first_condition, second_condition,
                                     crossover_strat)
        self._enforce_valid_genotype(first_condition.genotype)
        self._enforce_valid_genotype(second_condition.genotype)

    def _enforce_valid_genotype(self, genotype):
        self._enforce_genotype_is_inside_situation_space(genotype)

    def _enforce_genotype_is_inside_situation_space(self, genotype):
        for lower_idx in range(0, len(genotype), 2):
            frac_to_upper_idx = lower_idx + 1
            lower_allele = genotype[lower_idx]
            frac_to_upper_allele = genotype[frac_to_upper_idx]
            situation_space_idx = int(lower_idx / 2)
            self._truncate_allele_vals_if_needed(lower_allele,
                                                 frac_to_upper_allele,
                                                 situation_space_idx)

    def _truncate_allele_vals_if_needed(self, lower_allele,
                                        frac_to_upper_allele,
                                        situation_space_idx):
        dimension = self._situation_space[situation_space_idx]
        upper = self._calc_upper(lower_allele.value,
                                 frac_to_upper_allele.value, dimension.upper)

        # truncate lower
        trunc_lower = max(lower_allele.value, dimension.lower)
        lower_allele.value = trunc_lower

        # truncate upper
        trunc_upper = min(upper, dimension.upper)

        # re-calc frac to upper
        frac_to_upper_allele.value = self._calc_frac_to_upper(
            trunc_lower, trunc_upper, dimension.upper)

    def mutate_condition(self, condition, situation=None):
        genotype = condition.genotype
        for allele in genotype:
            self._mutate_allele(allele)
        self._enforce_valid_genotype(genotype)

    def _mutate_allele(self, allele):
        should_mutate = get_rng().rand() < get_hyperparam("mu")
        if should_mutate:
            adjustment_magnitude = get_rng().uniform(0, get_hyperparam("m"))
            adjustment_sign = get_rng().choice([1, -1])
            adjustment_amount = adjustment_magnitude * adjustment_sign
            allele += adjustment_amount

    def _make_phenotype_interval_from_allele_vals(self, first_val, second_val,
                                                  situation_space_idx):
        lower = first_val
        frac_to_upper = second_val
        dimension = self._situation_space[situation_space_idx]
        upper = self._calc_upper(lower, frac_to_upper, dimension.upper)
        return Interval(lower, upper)

from piecewise.dtype import Condition, FloatAllele, Genotype
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng

from .interval import Interval
from .interval_rule_repr import IntervalRuleReprABC


def make_ordered_bound_rule_repr(env):
    return OrderedBoundRuleRepr(situation_space=env.obs_space)


class OrderedBoundRuleRepr(IntervalRuleReprABC):
    """Rule representation that works with (lower, upper) interval
    predicates."""
    def __init__(self, situation_space):
        super().__init__(situation_space)

    def _gen_covering_condition(self, situation):
        alleles = []
        for situation_elem in situation:
            lower = situation_elem - get_rng().uniform(
                0, get_hyperparam("s_nought"))
            upper = situation_elem + get_rng().uniform(
                0, get_hyperparam("s_nought"))
            assert lower <= upper
            alleles.append(FloatAllele(lower))
            alleles.append(FloatAllele(upper))
        genotype = Genotype(alleles)
        self._enforce_valid_genotype(genotype)
        return Condition(genotype)

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        super().crossover_conditions(first_condition, second_condition,
                                     crossover_strat)
        self._enforce_valid_genotype(first_condition.genotype)
        self._enforce_valid_genotype(second_condition.genotype)

    def _enforce_valid_genotype(self, genotype):
        self._enforce_genotype_is_in_valid_order(genotype)
        self._enforce_genotype_is_inside_situation_space(genotype)

    def _enforce_genotype_is_in_valid_order(self, genotype):
        for lower_idx in range(0, len(genotype), 2):
            upper_idx = lower_idx + 1
            lower_allele = genotype[lower_idx]
            upper_allele = genotype[upper_idx]
            if lower_allele > upper_allele:
                # swap lower and upper
                genotype[lower_idx], genotype[upper_idx] = \
                        genotype[upper_idx], genotype[lower_idx]
            assert genotype[lower_idx] <= genotype[upper_idx]

    def _enforce_genotype_is_inside_situation_space(self, genotype):
        for lower_idx in range(0, len(genotype), 2):
            upper_idx = lower_idx + 1
            lower_allele = genotype[lower_idx]
            upper_allele = genotype[upper_idx]
            assert lower_allele <= upper_allele
            situation_space_idx = int(lower_idx / 2)
            self._truncate_allele_vals_if_needed(lower_allele, upper_allele,
                                                 situation_space_idx)

    def _truncate_allele_vals_if_needed(self, lower_allele, upper_allele,
                                        situation_space_idx):
        dimension = self._situation_space[situation_space_idx]
        lower_allele.value = max(lower_allele.value, dimension.lower)
        upper_allele.value = min(upper_allele.value, dimension.upper)
        assert lower_allele.value >= dimension.lower
        assert upper_allele.value <= dimension.upper

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

    def _make_phenotye_interval_from_allele_vals(self,
                                                 first_val,
                                                 second_val,
                                                 situation_space_idx=None):
        lower = first_val
        upper = second_val
        return Interval(lower, upper)

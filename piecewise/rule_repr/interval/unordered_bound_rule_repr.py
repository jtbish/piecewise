from piecewise.dtype import Condition, FloatAllele, Genotype
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng

from .interval import Interval
from .interval_rule_repr import IntervalRuleReprABC


def make_unordered_bound_rule_repr(env):
    return UnorderedBoundRuleRepr(situation_space=env.obs_space)


class UnorderedBoundRuleRepr(IntervalRuleReprABC):
    """Rule representation that works with (first, second) interval
    predicates."""
    def __init__(self, situation_space):
        super().__init__(situation_space)

    def _gen_covering_condition(self, situation):
        alleles = []
        for situation_elem in situation:
            a = situation_elem - get_rng().uniform(0,
                                                   get_hyperparam("s_nought"))
            b = situation_elem + get_rng().uniform(0,
                                                   get_hyperparam("s_nought"))
            # flip a coin to determine if a or b is first
            if get_rng().rand() < 0.5:
                # a first
                alleles.append(FloatAllele(a))
                alleles.append(FloatAllele(b))
            else:
                # b first
                alleles.append(FloatAllele(b))
                alleles.append(FloatAllele(a))
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
        self._enforce_genotype_is_inside_situation_space(genotype)

    def _enforce_genotype_is_inside_situation_space(self, genotype):
        for first_idx in range(0, len(genotype), 2):
            second_idx = first_idx + 1
            first_allele = genotype[first_idx]
            second_allele = genotype[second_idx]
            situation_space_idx = int(first_idx / 2)
            self._truncate_allele_vals_if_needed(first_allele, second_allele,
                                                 situation_space_idx)

    def _truncate_allele_vals_if_needed(self, first_allele, second_allele,
                                        situation_space_idx):
        dimension = self._situation_space[situation_space_idx]
        if first_allele <= second_allele:
            # first allele is lower bound, second allele upper bound
            first_allele.value = max(first_allele.value, dimension.lower)
            second_allele.value = min(second_allele.value, dimension.upper)
        else:
            # first allele is upper bound, second allele lower bound
            first_allele.value = min(first_allele.value, dimension.upper)
            second_allele.value = max(second_allele.value, dimension.lower)

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

    def _make_phenotype_interval_from_allele_vals(self,
                                                  first_val,
                                                  second_val,
                                                  situation_space_idx=None):
        lower = min(first_val, second_val)
        upper = max(first_val, second_val)
        return Interval(lower, upper)

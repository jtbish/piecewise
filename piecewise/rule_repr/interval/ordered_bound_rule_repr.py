from piecewise.dtype import Condition
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng

from .interval_rule_repr import IntervalRuleReprABC
from .predicate.ordered_bound_predicate import OrderedBoundPredicate


def make_ordered_bound_rule_repr(env):
    return OrderedBoundRuleRepr(situation_space=env.obs_space,
                                env_dtype=env.dtype)


class OrderedBoundRuleRepr(IntervalRuleReprABC):
    """Rule representation that works with (lower, upper) interval
    predicates."""
    def __init__(self, situation_space, env_dtype):
        super().__init__(situation_space, env_dtype)

    def _create_wildcard_predicate_for_dim(self, dimension):
        lower = dimension.lower
        upper = dimension.upper
        return self._make_predicate(lower, upper)

    def _make_predicate(self, lower, upper):
        return OrderedBoundPredicate(lower, upper)

    def gen_covering_condition(self, situation):
        alleles = []
        for (idx, situation_elem) in enumerate(situation):
            lower = situation_elem - get_rng().uniform(
                0, get_hyperparam("s_nought"))
            upper = situation_elem + get_rng().uniform(
                0, get_hyperparam("s_nought"))
            assert lower <= upper
            alleles.append(self._allele_type(lower))
            alleles.append(self._allele_type(upper))
        condition = Condition(rule_repr=self, alleles=alleles)
        self._enforce_alleles_are_inside_situation_space(condition)
        return condition

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        super().crossover_conditions(first_condition, second_condition,
                                     crossover_strat)
        self._enforce_alleles_are_in_valid_order(first_condition)
        self._enforce_alleles_are_in_valid_order(second_condition)

    def mutate_condition(self, condition, situation=None):
        for allele in condition:
            self._mutate_allele(allele)
        self._enforce_alleles_are_in_valid_order(condition)
        self._enforce_alleles_are_inside_situation_space(condition)

    def _mutate_allele(self, allele):
        should_mutate = get_rng().rand() < get_hyperparam("mu")
        if should_mutate:
            adjustment_magnitude = get_rng().uniform(0, get_hyperparam("m"))
            adjustment_sign = get_rng().choice([1, -1])
            adjustment_amount = adjustment_magnitude * adjustment_sign
            allele += adjustment_amount

    def _enforce_alleles_are_in_valid_order(self, condition):
        assert len(condition) % 2 == 0
        for lower_idx in range(0, len(condition), 2):
            upper_idx = lower_idx + 1
            lower_allele = condition[lower_idx]
            upper_allele = condition[upper_idx]
            if lower_allele > upper_allele:
                # swap lower and upper
                condition[lower_idx], condition[upper_idx] = \
                        condition[upper_idx], condition[lower_idx]
            assert condition[lower_idx] <= condition[upper_idx]

    def _enforce_alleles_are_inside_situation_space(self, condition):
        assert len(condition) % 2 == 0
        for lower_idx in range(0, len(condition), 2):
            upper_idx = lower_idx + 1
            lower_allele = condition[lower_idx]
            upper_allele = condition[upper_idx]
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

    def genotype_to_phenotype(self, condition):
        assert len(condition) % 2 == 0
        phenotype = []
        for lower_idx in range(0, len(condition), 2):
            upper_idx = lower_idx + 1
            lower = condition[lower_idx].value
            upper = condition[upper_idx].value
            phenotype.append(self._make_predicate(lower, upper))
        return tuple(phenotype)

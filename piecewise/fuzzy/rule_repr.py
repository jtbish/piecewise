import numpy as np

from piecewise.dtype import DataSpaceBuilder, Dimension
from piecewise.rule_repr import DiscereteMinSpanRuleRepr
from piecewise.rule_repr.rule_repr import IRuleRepr

from .condition import MIN_MATCHING_DEGREE


class FuzzyMinSpanRuleRepr(IRuleRepr):
    """Pretty much the main diff between this and MSR is that there is no
    situation space, only ling vars with their corresponding fuzzy sets."""
    def __init__(self, ling_vars, logical_or_strat, logical_and_strat):
        situation_space = \
            self._build_wrapped_situation_space_from_ling_vars(ling_vars)
        self._wrapped_msr = DiscereteMinSpanRuleRepr(situation_space)
        self._ling_vars = tuple(ling_vars)
        self._logical_or_strat = logical_or_strat
        self._logical_and_strat = logical_and_strat

    def _build_wrapped_situation_space_from_ling_vars(self, ling_vars):
        situation_space_builder = DataSpaceBuilder()
        for ling_var in ling_vars:
            dim_upper = (len(ling_var.membership_funcs) - 1)
            dim = Dimension(lower=0, upper=dim_upper)
            situation_space_builder.add_dim(dim)
        return situation_space_builder.create_space()

    def does_match(self, condition, situation):
        """Matching needs to compute the truth degree of the condition given
        the situation, then if truth degree is > 0.0 it matches."""
        matching_degree = self._eval_condition(condition, situation)
        condition.matching_degree = matching_degree
        return matching_degree > MIN_MATCHING_DEGREE

    def _eval_condition(self, condition, situation):
        assert len(situation) == len(self._ling_vars)
        phenotype = self.map_genotype_to_phenotype(condition.genotype)
        ling_var_ress = []
        for (situation_elem, phenotype_interval, ling_var) in \
                zip(situation, phenotype, self._ling_vars):
            ling_var_ress.append(
                self._eval_phenotype_interval(situation_elem,
                                              phenotype_interval, ling_var))
        assert len(ling_var_ress) == len(self._ling_vars)
        return self._logical_and_strat(ling_var_ress)

    def _eval_phenotype_interval(self, situation_elem, phenotype_interval,
                                 ling_var):
        active_membership_func_idxs = \
            list(range(phenotype_interval.lower, (phenotype_interval.upper+1)))
        membership_func_ress = \
            ling_var.eval_membership_funcs(active_membership_func_idxs,
                                           situation_elem)
        return self._logical_or_strat(membership_func_ress)

    def gen_covering_condition(self, situation):
        situation_for_wrapped = \
            self._create_covering_situation_for_wrapped(situation)
        return self._wrapped_msr.gen_covering_condition(
            situation=situation_for_wrapped)

    def _create_covering_situation_for_wrapped(self, situation):
        result = []
        for (ling_var, situation_elem) in zip(self._ling_vars, situation):
            result.append(
                self._find_best_matching_member_func_idx(
                    ling_var, situation_elem))
        return tuple(result)

    def _find_best_matching_member_func_idx(self, ling_var, situation_elem):
        membership_func_ress = \
            ling_var.eval_all_membership_funcs(situation_elem)
        return np.argmax(membership_func_ress)

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        return self._wrapped_msr.crossover_conditions(first_condition,
                                                      second_condition,
                                                      crossover_strat)

    def mutate_condition(self, condition, situation=None):
        self._wrapped_msr.mutate_condition(condition, situation=situation)

    def calc_generality(self, condition):
        return self._wrapped_msr.calc_generality(condition)

    def check_condition_subsumption(self, first_condition, second_condition):
        return self._wrapped_msr.check_condition_subsumption(
            first_condition, second_condition)

    def map_genotype_to_phenotype(self, genotype):
        return self._wrapped_msr.map_genotype_to_phenotype(genotype)

    def calc_max_matching_degree(self, situation):
        ling_var_ress = []
        for (situation_elem, ling_var) in zip(situation, self._ling_vars):
            ling_var_res = \
                self._logical_or_strat(
                        ling_var.eval_all_membership_funcs(situation_elem))
            ling_var_ress.append(ling_var_res)
        return self._logical_and_strat(ling_var_ress)

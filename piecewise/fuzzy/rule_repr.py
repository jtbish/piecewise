import abc

import numpy as np

from piecewise.dtype import Condition, DataSpaceBuilder, Dimension, Genotype
from piecewise.dtype.config import float_bounds_tol
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng
from piecewise.rule_repr import DiscereteMinSpanRuleRepr, IRuleRepr
from piecewise.util import truncate_val

MIN_MATCHING_DEGREE = 0.0
MAX_MATCHING_DEGREE = 1.0


class FuzzyRuleReprABC(IRuleRepr, metaclass=abc.ABCMeta):
    def __init__(self, ling_vars):
        self._ling_vars = tuple(ling_vars)

    def does_match(self, condition, situation):
        """Matching needs to compute the truth degree of the condition given
        the situation, then if truth degree is > 0.0 it matches."""
        matching_degree = self.eval_condition(condition, situation)
        return matching_degree > MIN_MATCHING_DEGREE

    def eval_condition(self, condition, situation):
        matching_degree = self._eval_condition(condition, situation)
        assert (MIN_MATCHING_DEGREE - float_bounds_tol) <= matching_degree \
            <= (MAX_MATCHING_DEGREE + float_bounds_tol)
        return matching_degree

    @abc.abstractmethod
    def _eval_condition(self, condition, situation):
        raise NotImplementedError

    @abc.abstractmethod
    def gen_covering_condition(self, situation):
        raise NotImplementedError

    @abc.abstractmethod
    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        raise NotImplementedError

    @abc.abstractmethod
    def mutate_condition(self, condition, situation=None):
        raise NotImplementedError

    @abc.abstractmethod
    def calc_generality(self, condition):
        raise NotImplementedError

    @abc.abstractmethod
    def check_condition_subsumption(self, first_condition, second_condition):
        raise NotImplementedError

    @abc.abstractmethod
    def map_genotype_to_phenotype(self, genotype):
        raise NotImplementedError

    @abc.abstractmethod
    def calc_max_matching_degree(self, situation):
        raise NotImplementedError

    def _find_best_matching_member_func_idx(self, ling_var, situation_elem):
        membership_func_ress = \
            ling_var.eval_all_membership_funcs(situation_elem)
        return np.argmax(membership_func_ress)


class FuzzyMinSpanRuleRepr(FuzzyRuleReprABC):
    """Pretty much the main diff between this and MSR is that there is no
    situation space, only ling vars with their corresponding fuzzy sets."""
    def __init__(self, ling_vars, logical_or_strat, logical_and_strat):
        super().__init__(ling_vars)
        self._logical_or_strat = logical_or_strat
        self._logical_and_strat = logical_and_strat
        situation_space = \
            self._build_wrapped_situation_space_from_ling_vars(ling_vars)
        self._wrapped_msr = DiscereteMinSpanRuleRepr(situation_space)

    def _build_wrapped_situation_space_from_ling_vars(self, ling_vars):
        situation_space_builder = DataSpaceBuilder()
        for ling_var in ling_vars:
            dim_upper = (len(ling_var.membership_funcs) - 1)
            dim = Dimension(lower=0, upper=dim_upper)
            situation_space_builder.add_dim(dim)
        return situation_space_builder.create_space()

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
        return \
            self._wrapped_msr.gen_covering_condition(
                situation=situation_for_wrapped)

    def _create_covering_situation_for_wrapped(self, situation):
        result = []
        for (ling_var, situation_elem) in zip(self._ling_vars, situation):
            result.append(
                self._find_best_matching_member_func_idx(
                    ling_var, situation_elem))
        return tuple(result)

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        self._wrapped_msr.crossover_conditions(first_condition,
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


class FuzzyConjunctiveRuleRepr(FuzzyRuleReprABC):
    def __init__(self, ling_vars, logical_and_strat):
        super().__init__(ling_vars)
        self._logical_and_strat = logical_and_strat

    def _eval_condition(self, condition, situation):
        assert len(situation) == len(self._ling_vars)
        phenotype = self.map_genotype_to_phenotype(condition.genotype)
        ling_var_ress = []
        for (situation_elem, phenotype_elem, ling_var) in \
                zip(situation, phenotype, self._ling_vars):
            ling_var_ress.append(
                self._eval_phenotype_elem(situation_elem, phenotype_elem,
                                          ling_var))
        assert len(ling_var_ress) == len(self._ling_vars)
        return self._logical_and_strat(ling_var_ress)

    def _eval_phenotype_elem(self, situation_elem, phenotype_elem, ling_var):
        active_idx = phenotype_elem
        membership_res = ling_var.eval_membership_func(active_idx,
                                                       situation_elem)
        return membership_res

    def gen_covering_condition(self, situation):
        alleles = []
        for (ling_var, situation_elem) in zip(self._ling_vars, situation):
            alleles.append(
                self._find_best_matching_member_func_idx(
                    ling_var, situation_elem))
        genotype = Genotype(alleles)
        return Condition(genotype)

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        crossover_strat(first_condition.genotype, second_condition.genotype)

    def mutate_condition(self, condition, situation=None):
        genotype = condition.genotype
        for allele_idx in range(0, len(genotype)):
            should_mutate = get_rng().rand() < get_hyperparam("mu")
            if should_mutate:
                # mutation draws from +-[0, m_nought]
                m_nought = get_hyperparam("m_nought")
                assert m_nought > 0
                mut_choices = range(0, m_nought + 1)
                mutation_magnitude = get_rng().choice(mut_choices)
                mutation_sign = get_rng().choice([1, -1])
                mutation_amount = mutation_magnitude * mutation_sign
                genotype[allele_idx] += mutation_amount
        self._enforce_genotype_maps_to_valid_phenotype(genotype)

    def _enforce_genotype_maps_to_valid_phenotype(self, genotype):
        assert len(genotype) == len(self._ling_vars)
        for (allele_idx, ling_var) in zip(range(0, len(genotype)),
                                          self._ling_vars):
            allele = genotype[allele_idx]
            min_val = 0
            max_val = (len(ling_var.membership_funcs) - 1)
            allele = truncate_val(allele,
                                  lower_bound=min_val,
                                  upper_bound=max_val)
            genotype[allele_idx] = allele

    def calc_generality(self, condition):
        raise NotImplementedError  # should never get here

    def check_condition_subsumption(self, first_condition, second_condition):
        raise NotImplementedError  # should never get here

    def map_genotype_to_phenotype(self, genotype):
        return tuple([allele for allele in genotype])

    def calc_max_matching_degree(self, situation):
        ling_var_ress = []
        for (situation_elem, ling_var) in zip(situation, self._ling_vars):
            ling_var_res = max(
                ling_var.eval_all_membership_funcs(situation_elem))
            ling_var_ress.append(ling_var_res)
        return self._logical_and_strat(ling_var_ress)

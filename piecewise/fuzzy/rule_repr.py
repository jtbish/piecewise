import abc
import copy
import logging
import itertools

import numpy as np

from piecewise.dtype import (Condition, DataSpaceBuilder, Dimension, Genotype,
                            Population, Rule)
from piecewise.dtype.config import float_bounds_tol
from piecewise.error.core_errors import InternalError
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng
from piecewise.rule_repr import DiscereteMinSpanRuleRepr, IRuleRepr
from piecewise.util import truncate_val
from piecewise.constants import TIME_STEP_MIN

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

    @abc.abstractmethod
    def gen_complete_population(self, env_action_set, classifier_factory):
        raise NotImplementedError


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
            ling_var_res = max(
                ling_var.eval_all_membership_funcs(situation_elem))
            ling_var_ress.append(ling_var_res)
        return self._logical_and_strat(ling_var_ress)

    def gen_complete_population(self, env_action_set, classifier_factory):
        raise NotImplementedError


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

    def gen_complete_population(self, env_action_set, classifier_factory):
        # construct all possible rules with time step = 0

        possible_dim_alleles = []
        for ling_var in self._ling_vars:
            possible_dim_alleles.append(list(
                range(0, ling_var.num_membership_funcs)))
        condition_alleles = list(itertools.product(*possible_dim_alleles))

        max_pop_micros = np.product([ling_var.num_membership_funcs for
            ling_var in self._ling_vars])*len(env_action_set)
        population = Population(max_micros=max_pop_micros)

        for alleles in condition_alleles:
            for action in env_action_set:
                genotype = Genotype(alleles)
                condition = Condition(genotype)
                rule = Rule(condition, action,
                        num_features=len(self._ling_vars))
                classifier = classifier_factory(rule, time_step=TIME_STEP_MIN)
                population.add(classifier)

        assert population.num_micros == max_pop_micros
        assert population.num_micros == population.num_macros
        return population


class FuzzyCNFRuleRepr(FuzzyRuleReprABC):
    def __init__(self, ling_vars, logical_or_strat, logical_and_strat):
        super().__init__(ling_vars)
        self._logical_or_strat = logical_or_strat
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
        # phenotype elem is tuple of active membership func idxs
        assert len(phenotype_elem) >= 1
        membership_ress = []
        for active_idx in phenotype_elem:
            membership_ress.append(
                ling_var.eval_membership_func(active_idx, situation_elem))
        return self._logical_or_strat(membership_ress)

    def gen_covering_condition(self, situation):
        alleles = []
        for (ling_var, situation_elem) in zip(self._ling_vars, situation):
            best_matching_idx = self._find_best_matching_member_func_idx(
                ling_var, situation_elem)
            alleles_for_ling_var = []
            for idx in range(0, ling_var.num_membership_funcs):
                if idx == best_matching_idx:
                    alleles_for_ling_var.append(1)
                else:
                    alleles_for_ling_var.append(0)
            # only one allele should be active for each ling var
            assert alleles_for_ling_var.count(1) == 1
            alleles.extend(alleles_for_ling_var)
        assert len(alleles) == sum(
            [ling_var.num_membership_funcs for ling_var in self._ling_vars])
        genotype = Genotype(alleles)
        return Condition(genotype)

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        first_genotype_before_crossover = copy.deepcopy(first_condition.genotype)
        second_genotype_before_crossover = copy.deepcopy(second_condition.genotype)

        crossover_strat(first_condition.genotype, second_condition.genotype)

        self._correct_crossover_res_if_necessary(first_genotype_before_crossover,
                first_condition.genotype)
        self._correct_crossover_res_if_necessary(second_genotype_before_crossover,
                second_condition.genotype)

        self._assert_genotype_is_valid(first_condition.genotype)
        self._assert_genotype_is_valid(second_condition.genotype)

    def _correct_crossover_res_if_necessary(self, genotype_before,
            genotype_after):
        ling_var_genotype_ranges = self._get_ling_var_genotype_ranges()
        for (start_idx, num_alleles) in ling_var_genotype_ranges:
            end_idx_exclusive = (start_idx + num_alleles)
            after_ling_var_alleles = genotype_after[start_idx:end_idx_exclusive]
            has_no_ones_after = after_ling_var_alleles.count(1) == 0
            if has_no_ones_after:
                one_allele_idxs_before = [
                    idx for idx in range(start_idx, end_idx_exclusive)
                    if genotype_before[idx] == 1
                ]
                assert len(one_allele_idxs_before) >= 1
                idx_for_one = get_rng().choice(one_allele_idxs_before)
                genotype_after[idx_for_one] = 1

    def _get_ling_var_genotype_ranges(self):
        ranges = []
        range_start_idx = 0
        for ling_var in self._ling_vars:
            num_alleles = ling_var.num_membership_funcs
            ranges.append((range_start_idx, num_alleles))
            range_start_idx += num_alleles
        assert range_start_idx == sum(
            [ling_var.num_membership_funcs for ling_var in self._ling_vars])
        return tuple(ranges)

    def _assert_genotype_is_valid(self, genotype):
        ling_var_genotype_ranges = self._get_ling_var_genotype_ranges()
        for ling_var_genotype_range in ling_var_genotype_ranges:
            ling_var_alleles = self._get_ling_var_alleles(genotype,
                    ling_var_genotype_range)
            has_ones = ling_var_alleles.count(1) >= 1
            assert has_ones, f"{ling_var_alleles}"

    def mutate_condition(self, condition, situation=None):
        should_do_mutation = get_rng().rand() < get_hyperparam("mu")
        if should_do_mutation:
            self._mutate_condition(condition)
            self._assert_genotype_is_valid(condition.genotype)

    def _mutate_condition(self, condition):
        ling_var_genotype_ranges = self._get_ling_var_genotype_ranges()
        ling_var_idx_to_mut = get_rng().choice(range(len(self._ling_vars)))
        ling_var_genotype_range = ling_var_genotype_ranges[ling_var_idx_to_mut]

        mut_strat = self._choose_mut_strat_for_ling_var(
            condition.genotype, ling_var_genotype_range)
        if mut_strat == "expand":
            self._mut_expand(condition.genotype, ling_var_genotype_range)
        elif mut_strat == "contract":
            self._mut_contract(condition.genotype,
                               ling_var_genotype_range)
        elif mut_strat == "shift":
            self._mut_shift(condition.genotype, ling_var_genotype_range)
        else:
            raise InternalError("Should not get here")

    def _choose_mut_strat_for_ling_var(self, genotype,
                                       ling_var_genotype_range):
        ling_var_alleles = self._get_ling_var_alleles(genotype,
                                                      ling_var_genotype_range)

        # can always shift because it needs at least a single one
        # allele present to operate and guarantees that at least a single one
        # allele remains afterwards
        possible_strats = ["shift"]
        # need at least a single zero allele to expand
        could_expand = ling_var_alleles.count(0) >= 1
        if could_expand:
            possible_strats.append("expand")
        # need at least two one alleles to contract so at least a single one
        # allele remains
        could_contract = ling_var_alleles.count(1) >= 2
        if could_contract:
            possible_strats.append("contract")

        mut_strat = get_rng().choice(possible_strats)
        return mut_strat

    def _get_ling_var_alleles(self, genotype, ling_var_genotype_range):
        (start_idx, num_alleles) = ling_var_genotype_range
        return genotype[start_idx:(start_idx + num_alleles)]

    def _mut_expand(self, genotype, ling_var_genotype_range):
        logging.debug("Mut expand")
        (start_idx, num_alleles) = ling_var_genotype_range
        end_idx_exclusive = (start_idx + num_alleles)
        # expansion flips a randomly chosen zero to a one
        zero_allele_idxs = [
            idx for idx in range(start_idx, end_idx_exclusive)
            if genotype[idx] == 0
        ]
        assert len(zero_allele_idxs) >= 1
        idx_to_flip = get_rng().choice(zero_allele_idxs)
        genotype[idx_to_flip] = 1

    def _mut_contract(self, genotype, ling_var_genotype_range):
        logging.debug("Mut contract")
        (start_idx, num_alleles) = ling_var_genotype_range
        end_idx_exclusive = (start_idx + num_alleles)
        # contraction flips a randomly chosen one to a zero
        one_allele_idxs = [
            idx for idx in range(start_idx, end_idx_exclusive)
            if genotype[idx] == 1
        ]
        assert len(one_allele_idxs) >= 2
        idx_to_flip = get_rng().choice(one_allele_idxs)
        genotype[idx_to_flip] = 0

    def _mut_shift(self, genotype, ling_var_genotype_range):
        logging.debug("Mut shift")
        (start_idx, num_alleles) = ling_var_genotype_range
        end_idx_exclusive = (start_idx + num_alleles)
        # shift flips a randomly chosen one to a zero then
        # sets either the allele before or after to a one depending on what
        # is possible
        one_allele_idxs = [
            idx for idx in range(start_idx, end_idx_exclusive)
            if genotype[idx] == 1
        ]
        assert len(one_allele_idxs) >= 1
        idx_to_flip = get_rng().choice(one_allele_idxs)
        genotype[idx_to_flip] = 0

        # get info about adjacent alleles of the ling var
        prev_idx = max(start_idx, (idx_to_flip - 1))
        end_idx_inclusive = (end_idx_exclusive - 1)
        next_idx = min(end_idx_inclusive, (idx_to_flip + 1))

        # pick an adjacent allele to set to 1
        if prev_idx == start_idx:
            # edge case, allele flipped to zero originally is first one in seq.
            # for ling var, so set the allele to the right of it to one
            idx_for_one = next_idx
        elif next_idx == end_idx_inclusive:
            # edge case, allele flipped to zero originally is last one in seq.
            # for ling var, so set the allele to the left of it to one
            idx_for_one = prev_idx
        else:
            # nominal case, allele flipped to zero originally not at
            # boundaries, pick either prev or next allele at random to set to
            # one
            idx_for_one = get_rng().choice([prev_idx, next_idx])
        genotype[idx_for_one] = 1

    def calc_generality(self, condition):
        genotype = condition.genotype
        generality = genotype.count(1) / len(genotype)
        # 0.0 < is not a mistake as cannot have no occurences of 1 in the
        # genotype
        assert 0.0 < generality <= 1.0
        return generality

    def check_condition_subsumption(self, first_condition, second_condition):
        for (first_allele, second_allele) in zip(first_condition.genotype,
                                                 second_condition.genotype):
            if first_allele == 0 and second_allele == 1:
                return False
        return True

    def map_genotype_to_phenotype(self, genotype):
        ling_var_lens = [
            ling_var.num_membership_funcs for ling_var in self._ling_vars
        ]
        allele_idx = 0
        phenotype = []
        for ling_var_len in ling_var_lens:
            phenotype_elem = []
            for membership_func_idx in range(0, ling_var_len):
                if genotype[allele_idx] == 1:
                    phenotype_elem.append(membership_func_idx)
                allele_idx += 1
            assert len(phenotype_elem) >= 1
            phenotype.append(tuple(phenotype_elem))
        assert allele_idx == len(genotype)
        return tuple(phenotype)

    def calc_max_matching_degree(self, situation):
        ling_var_ress = []
        for (situation_elem, ling_var) in zip(situation, self._ling_vars):
            ling_var_res = max(
                ling_var.eval_all_membership_funcs(situation_elem))
            ling_var_ress.append(ling_var_res)
        return self._logical_and_strat(ling_var_ress)

    def gen_complete_population(self, env_action_set, classifier_factory):
        raise NotImplementedError

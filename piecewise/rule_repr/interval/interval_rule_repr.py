import abc

from ..rule_repr import IRuleRepr
from .interval import Interval


class IntervalRuleReprABC(IRuleRepr, metaclass=abc.ABCMeta):
    def __init__(self, situation_space):
        self._situation_space = situation_space
        self._wildcard_intervals = self._create_wildcard_intervals()

    def _create_wildcard_intervals(self):
        return tuple([
            self._create_wildcard_interval_for_dim(dimension)
            for dimension in self._situation_space
        ])

    def _create_wildcard_interval_for_dim(self, dimension):
        return Interval(dimension.lower, dimension.upper)

    def does_match(self, condition, situation):
        phenotype = condition.phenotype(self)
        for (interval, situation_elem) in zip(phenotype, situation):
            if not interval.contains_point(situation_elem):
                return False
        return True

    def gen_covering_condition(self, situation):
        condition = self._gen_covering_condition(situation)
        genotype_has_even_len = len(condition.genotype) % 2 == 0
        assert genotype_has_even_len
        return condition

    @abc.abstractmethod
    def _gen_covering_condition(self, situation):
        raise NotImplementedError

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        crossover_strat(first_condition.genotype, second_condition.genotype)

    @abc.abstractmethod
    def mutate_condition(self, condition, situation=None):
        raise NotImplementedError

    def _is_wildcard(self, interval_predicate, phenotype_idx):
        return interval_predicate.contains(
            self._wildcard_predicates[phenotype_idx])

    def map_genotype_to_phenotype(self, genotype):
        assert len(genotype) % 2 == 0
        phenotype = []
        for first_allele_idx in range(0, len(genotype), 2):
            second_allele_idx = first_allele_idx + 1
            first_val = genotype[first_allele_idx].value
            second_val = genotype[second_allele_idx].value
            situation_space_idx = int(first_allele_idx / 2)
            phenotype.append(
                self._make_phenotype_interval_from_allele_vals(
                    first_val, second_val, situation_space_idx))
        return tuple(phenotype)

    @abc.abstractmethod
    def _make_phenotype_interval_from_allele_vals(self, first_val, second_val,
                                                  situation_space_idx):
        raise NotImplementedError

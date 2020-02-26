import abc

from ..rule_repr import IRuleRepr
from .interval import Interval


class IntervalRuleReprABC(IRuleRepr, metaclass=abc.ABCMeta):
    def __init__(self, situation_space):
        self._situation_space = situation_space
        self._wildcard_intervals = self._create_wildcard_intervals()

    def _create_wildcard_intervals(self):
        return tuple([
            Interval(dimension.lower, dimension.upper)
            for dimension in self._situation_space
        ])

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

    def _is_wildcard(self, interval, phenotype_idx):
        return interval.contains_interval(
            self._wildcard_intervals[phenotype_idx])

    def calc_generality(self, condition):
        phenotype = condition.phenotype(self)
        assert len(phenotype) == len(self._situation_space)
        cover_fractions = \
            [phenotype_interval.fraction_covered_of(wildcard_interval) for
                (phenotype_interval, wildcard_interval) in
                zip(phenotype, self._wildcard_intervals)]
        generality = sum(cover_fractions) / len(phenotype)
        assert 0.0 <= generality <= 1.0
        return generality

    def check_condition_subsumption(self, subsumer_condition,
                                    subsumee_condition):
        subsumer_phenotype = subsumer_condition.phenotype(self)
        subsumee_phenotype = subsumee_condition.phenotype(self)
        for (idx, subsumer_interval, subsumee_interval) in \
                enumerate(zip(subsumer_phenotype, subsumee_phenotype)):
            if not self._is_wildcard(subsumer_interval, idx) and \
                    subsumer_interval.contains_interval(subsumee_interval):
                return False
        return True

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

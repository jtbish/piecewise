from piecewise.dtype import Condition, Genotype
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng
from piecewise.util import truncate_val

from ..rule_repr import IRuleRepr
from .interval import ContinuousInterval


def make_continuous_min_percentage_rule_repr(env):
    return ContinuousMinPercentageRuleRepr(situation_space=env.obs_space)


class ContinuousMinPercentageRuleRepr(IRuleRepr):
    """Rule representation that works with (lower, frac_to_upper)
    genotype in continuous space."""
    _MIN_FRAC_TO_UPPER_VAL = 0.0
    _MAX_FRAC_TO_UPPER_VAL = 1.0

    def __init__(self, situation_space):
        self._situation_space = situation_space
        self._wildcard_intervals = \
            self._create_wildcard_intervals(self._situation_space)

    def _create_wildcard_intervals(self, situation_space):
        return tuple([
            ContinuousInterval(dimension.lower, dimension.upper)
            for dimension in situation_space
        ])

    def does_match(self, condition, situation):
        phenotype = condition.phenotype(self)
        for (interval, situation_elem) in zip(phenotype, situation):
            if not interval.contains_point(situation_elem):
                return False
        return True

    def gen_covering_condition(self, situation):
        alleles = []
        for (idx, situation_elem) in enumerate(situation):
            lower = situation_elem - get_rng().uniform(
                0, get_hyperparam("s_nought"))
            upper = situation_elem + get_rng().uniform(
                0, get_hyperparam("s_nought"))
            dimension = self._situation_space[idx]
            lower = truncate_val(lower,
                                 lower_bound=dimension.lower,
                                 upper_bound=dimension.upper)
            upper = truncate_val(upper,
                                 lower_bound=dimension.lower,
                                 upper_bound=dimension.upper)
            assert lower <= upper
            frac_to_upper = self._calc_frac_to_upper(lower, upper,
                                                     dimension.upper)
            alleles.append(lower)
            alleles.append(frac_to_upper)
        genotype = Genotype(alleles)
        return Condition(genotype)

    def _calc_frac_to_upper(self, lower, upper, dimension_upper):
        frac_to_upper = (upper - lower) / (dimension_upper - lower)
        assert 0.0 <= frac_to_upper <= 1.0
        return frac_to_upper

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        crossover_strat(first_condition.genotype, second_condition.genotype)
        self._enforce_genotype_maps_to_valid_phenotype(
            first_condition.genotype)
        self._enforce_genotype_maps_to_valid_phenotype(
            second_condition.genotype)

    def _enforce_genotype_maps_to_valid_phenotype(self, genotype):
        assert len(genotype) % 2 == 0
        self._truncate_lower_alleles(genotype)
        self._truncate_frac_to_upper_alleles(genotype)

    def _truncate_lower_alleles(self, genotype):
        for lower_allele_idx in range(0, len(genotype), 2):
            situation_space_idx = int(lower_allele_idx / 2)
            dimension = self._situation_space[situation_space_idx]
            lower_allele = genotype[lower_allele_idx]
            lower_allele = truncate_val(lower_allele,
                                        lower_bound=dimension.lower,
                                        upper_bound=dimension.upper)
            genotype[lower_allele_idx] = lower_allele

    def _truncate_frac_to_upper_alleles(self, genotype):
        for frac_to_upper_idx in range(1, len(genotype), 2):
            frac_to_upper_allele = genotype[frac_to_upper_idx]
            frac_to_upper_allele = \
                truncate_val(frac_to_upper_allele,
                             lower_bound=self._MIN_FRAC_TO_UPPER_VAL,
                             upper_bound=self._MAX_FRAC_TO_UPPER_VAL)
            genotype[frac_to_upper_idx] = frac_to_upper_allele

    def mutate_condition(self, condition, situation=None):
        genotype = condition.genotype
        for allele_idx in range(len(genotype)):
            should_mutate = get_rng().rand() < get_hyperparam("mu")
            if should_mutate:
                mutation_magnitude = get_rng().uniform(0, get_hyperparam("m"))
                mutation_sign = get_rng().choice([1, -1])
                mutation_amount = mutation_magnitude * mutation_sign
                genotype[allele_idx] += mutation_amount
        self._enforce_genotype_maps_to_valid_phenotype(genotype)

    def calc_generality(self, condition):
        phenotype = condition.phenotype(self)
        assert len(phenotype) == len(self._situation_space)
        cover_fractions = \
            [wildcard_interval.fraction_covered_by(phenotype_interval) for
                (wildcard_interval, phenotype_interval) in
                zip(self._wildcard_intervals, phenotype)]
        generality = sum(cover_fractions) / len(phenotype)
        assert 0.0 <= generality <= 1.0
        return generality

    def check_condition_subsumption(self, first_condition, second_condition):
        first_phenotype = first_condition.phenotype(self)
        second_phenotype = second_condition.phenotype(self)
        for idx, (first_interval, second_interval) in \
                enumerate(zip(first_phenotype, second_phenotype)):
            if not self._is_wildcard(first_interval, idx) and \
                    not first_interval.contains_interval(second_interval):
                return False
        return True

    def _is_wildcard(self, phenotype_interval, phenotype_idx):
        return phenotype_interval.contains_interval(
            self._wildcard_intervals[phenotype_idx])

    def map_genotype_to_phenotype(self, genotype):
        phenotype = []
        for lower_allele_idx in range(0, len(genotype), 2):
            frac_to_upper_idx = lower_allele_idx + 1
            lower_allele = genotype[lower_allele_idx]
            frac_to_upper_allele = genotype[frac_to_upper_idx]
            situation_space_idx = int(lower_allele_idx / 2)
            dimension = self._situation_space[situation_space_idx]
            interval = self._make_phenotype_interval_from_alleles(
                lower_allele, frac_to_upper_allele, dimension)
            assert dimension.lower <= interval.lower <= dimension.upper
            assert dimension.lower <= interval.upper <= dimension.upper
            phenotype.append(interval)
        return tuple(phenotype)

    def _make_phenotype_interval_from_alleles(self, lower_allele,
                                              frac_to_upper_allele, dimension):
        interval_lower = lower_allele
        interval_upper = self._calc_upper(lower_allele, frac_to_upper_allele,
                                          dimension.upper)
        return ContinuousInterval(interval_lower, interval_upper)

    def _calc_upper(self, lower, frac_to_upper, dimension_upper):
        span_from_lower = (dimension_upper - lower) * frac_to_upper
        upper = lower + span_from_lower
        return upper

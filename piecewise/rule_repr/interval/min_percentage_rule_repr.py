import abc

from piecewise.dtype import Condition, Genotype
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng
from piecewise.util import truncate_val

from ..rule_repr import IRuleRepr
from .interval import DiscreteInterval, ContinuousInterval


def make_continuous_min_percentage_rule_repr(env):
    return ContinuousMinPercentageRuleRepr(situation_space=env.obs_space)


def make_discrete_min_span_rule_repr(env):
    return DiscereteMinSpanRuleRepr(situation_space=env.obs_space)


class MinSpanRuleReprABC(IRuleRepr, metaclass=abc.ABCMeta):
    def __init__(self, situation_space, interval_cls):
        self._situation_space = situation_space
        self._interval_cls = interval_cls
        self._wildcard_intervals = \
            self._create_wildcard_intervals(self._situation_space,
                self._interval_cls)

    def _create_wildcard_intervals(self, situation_space, interval_cls):
        return tuple([
            interval_cls(dimension.lower, dimension.upper)
            for dimension in situation_space
        ])

    def does_match(self, condition, situation):
        phenotype = condition.phenotype(self)
        for (interval, situation_elem) in zip(phenotype, situation):
            if not interval.contains_point(situation_elem):
                return False
        return True

    @abc.abstractmethod
    def gen_covering_condition(self, situation):
        raise NotImplementedError

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        crossover_strat(first_condition.genotype, second_condition.genotype)
        self._enforce_genotype_maps_to_valid_phenotype(
            first_condition.genotype)
        self._enforce_genotype_maps_to_valid_phenotype(
            second_condition.genotype)

    @abc.abstractmethod
    def _enforce_genotype_maps_to_valid_phenotype(self, genotype):
        raise NotImplementedError

    def _truncate_lower_alleles(self, genotype):
        for lower_allele_idx in range(0, len(genotype), 2):
            situation_space_idx = int(lower_allele_idx / 2)
            dimension = self._situation_space[situation_space_idx]
            lower_allele = genotype[lower_allele_idx]
            lower_allele = truncate_val(lower_allele,
                                        lower_bound=dimension.lower,
                                        upper_bound=dimension.upper)
            genotype[lower_allele_idx] = lower_allele

    @abc.abstractmethod
    def mutate_condition(self, condition, situation=None):
        raise NotImplementedError

    @abc.abstractmethod
    def calc_generality(self, condition):
        raise NotImplementedError

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


class ContinuousMinPercentageRuleRepr(MinSpanRuleReprABC):
    """Rule representation that works with (lower, frac_to_upper)
    genotype in continuous space."""
    _MIN_FRAC_TO_UPPER_VAL = 0.0
    _MAX_FRAC_TO_UPPER_VAL = 1.0

    def __init__(self, situation_space):
        super().__init__(situation_space, interval_cls=ContinuousInterval)

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

    def _enforce_genotype_maps_to_valid_phenotype(self, genotype):
        assert len(genotype) % 2 == 0
        self._truncate_lower_alleles(genotype)
        self._truncate_frac_to_upper_alleles(genotype)

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
        assert len(phenotype) == len(self._wildcard_intervals)
        cover_fractions = \
            [wildcard_interval.fraction_covered_by(phenotype_interval) for
                (wildcard_interval, phenotype_interval) in
                zip(self._wildcard_intervals, phenotype)]
        generality = sum(cover_fractions) / len(phenotype)
        assert 0.0 <= generality <= 1.0
        return generality

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
        return self._interval_cls(interval_lower, interval_upper)

    def _calc_upper(self, lower, frac_to_upper, dimension_upper):
        span_from_lower = (dimension_upper - lower) * frac_to_upper
        upper = lower + span_from_lower
        return upper


class DiscereteMinSpanRuleRepr(MinSpanRuleReprABC):
    """Rule representation that works with (lower, span_to_upper)
    genotype in discrete space."""
    _MIN_SPAN_TO_UPPER_VAL = 0

    def __init__(self, situation_space):
        super().__init__(situation_space, interval_cls=DiscreteInterval)

    def gen_covering_condition(self, situation):
        alleles = []
        for (idx, situation_elem) in enumerate(situation):
            # covering draws from [0, s_nought]
            cover_choices = range(0, get_hyperparam("s_nought") + 1)
            lower = situation_elem - get_rng().choice(cover_choices)
            upper = situation_elem + get_rng().choice(cover_choices)
            dimension = self._situation_space[idx]
            lower = truncate_val(lower,
                                 lower_bound=dimension.lower,
                                 upper_bound=dimension.upper)
            upper = truncate_val(upper,
                                 lower_bound=dimension.lower,
                                 upper_bound=dimension.upper)
            assert lower <= upper
            span_to_upper = self._calc_span_to_upper(lower, upper, dimension)
            alleles.append(lower)
            alleles.append(span_to_upper)
        genotype = Genotype(alleles)
        return Condition(genotype)

    def _calc_span_to_upper(self, lower, upper, dimension):
        # span to upper is number of discrete steps to upper from lower,
        # NOT a fraction
        span_to_upper = upper - lower
        max_span_to_upper_val = dimension.upper - dimension.lower
        assert self._MIN_SPAN_TO_UPPER_VAL <= span_to_upper <= \
            max_span_to_upper_val
        return span_to_upper

    def _enforce_genotype_maps_to_valid_phenotype(self, genotype):
        assert len(genotype) % 2 == 0
        self._truncate_lower_alleles(genotype)
        self._truncate_span_to_upper_alleles(genotype)

    def _truncate_span_to_upper_alleles(self, genotype):
        for lower_allele_idx in range(0, len(genotype), 2):
            span_to_upper_allele_idx = lower_allele_idx + 1
            situation_space_idx = int(lower_allele_idx / 2)
            dimension = self._situation_space[situation_space_idx]

            lower_allele = genotype[lower_allele_idx]
            span_to_upper_allele = genotype[span_to_upper_allele_idx]
            max_span_to_upper_val = dimension.upper - lower_allele
            span_to_upper_allele = \
                truncate_val(span_to_upper_allele,
                             lower_bound=self._MIN_SPAN_TO_UPPER_VAL,
                             upper_bound=max_span_to_upper_val)
            genotype[span_to_upper_allele_idx] = span_to_upper_allele

    def mutate_condition(self, condition, situation=None):
        genotype = condition.genotype
        for allele_idx in range(len(genotype)):
            should_mutate = get_rng().rand() < get_hyperparam("mu")
            if should_mutate:
                # mutation draws from +-(0, m]
                mut_choices = range(1, get_hyperparam("m") + 1)
                mutation_magnitude = get_rng().choice(mut_choices)
                mutation_sign = get_rng().choice([1, -1])
                mutation_amount = mutation_magnitude * mutation_sign
                genotype[allele_idx] += mutation_amount
        self._enforce_genotype_maps_to_valid_phenotype(genotype)

    def calc_generality(self, condition):
        phenotype = condition.phenotype(self)
        assert len(phenotype) == len(self._wildcard_intervals)
        phenotype_interval_coverages = \
            [phenotype_interval.num_vals_covered() for phenotype_interval in
                phenotype]
        wildcard_interval_coverages = \
            [wildcard_interval.num_vals_covered() for wildcard_interval in
                self._wildcard_intervals]
        generality = sum(phenotype_interval_coverages) / \
            sum(wildcard_interval_coverages)
        assert 0.0 <= generality <= 1.0
        return generality

    def map_genotype_to_phenotype(self, genotype):
        phenotype = []
        for lower_allele_idx in range(0, len(genotype), 2):
            span_to_upper_idx = lower_allele_idx + 1
            lower_allele = genotype[lower_allele_idx]
            span_to_upper_allele = genotype[span_to_upper_idx]
            situation_space_idx = int(lower_allele_idx / 2)
            dimension = self._situation_space[situation_space_idx]
            interval = self._make_phenotype_interval_from_alleles(
                lower_allele, span_to_upper_allele)
            assert dimension.lower <= interval.lower <= dimension.upper
            assert dimension.lower <= interval.upper <= dimension.upper
            phenotype.append(interval)
        return tuple(phenotype)

    def _make_phenotype_interval_from_alleles(self, lower_allele,
                                              span_to_upper_allele):
        interval_lower = lower_allele
        interval_upper = lower_allele + span_to_upper_allele
        return self._interval_cls(interval_lower, interval_upper)

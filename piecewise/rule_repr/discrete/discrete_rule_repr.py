from piecewise.dtype import Condition, Genotype
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng

from ..rule_repr import IRuleRepr


class DiscreteRuleRepr(IRuleRepr):
    """Rule representation that works with discrete (i.e. integer) inputs,
    storing a single discrete value for each allele in the condition genotype.
    """
    _WILDCARD_ALLELE = "#"

    def does_match(self, condition, situation):
        """DOES MATCH function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        for (allele, situation_elem) in zip(condition.genotype, situation):
            allele_is_wildcard = self._is_wildcard(allele)
            allele_matches_input = allele == situation_elem
            if not allele_is_wildcard and \
                    not allele_matches_input:
                return False
        return True

    def gen_covering_condition(self, situation):
        """First part (condition generation) of
        GENERATE COVERING CLASSIFIER function from
        'An Algorithmic Description of XCS' (Butz and Wilson, 2002).
        """
        alleles = []
        for situation_elem in situation:
            should_make_wildcard = get_rng().rand() < get_hyperparam(
                "p_wildcard")
            if should_make_wildcard:
                alleles.append(self._WILDCARD_ALLELE)
            else:
                # copy situation
                alleles.append(situation_elem)
        genotype = Genotype(alleles)
        return Condition(genotype)

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        crossover_strat(first_condition.genotype, second_condition.genotype)

    def mutate_condition(self, condition, situation):
        """First part (condition mutation) of APPLY MUTATION function from 'An
        Algorithmic Description of XCS' (Butz and Wilson, 2002)."""
        genotype = condition.genotype
        for idx, (allele,
                  situation_elem) in enumerate(zip(genotype, situation)):
            should_mutate_allele = get_rng().rand() < get_hyperparam("mu")
            if should_mutate_allele:
                if self._is_wildcard(allele):
                    genotype[idx] = situation_elem
                else:
                    genotype[idx] = self._WILDCARD_ALLELE

    def _is_wildcard(self, allele):
        return allele == self._WILDCARD_ALLELE

    def calc_generality(self, condition):
        num_wildcards = [
            self._is_wildcard(allele) for allele in condition.genotype
        ].count(True)
        generality = num_wildcards / len(condition.genotype)
        assert 0.0 <= generality <= 1.0
        return generality

    def check_condition_subsumption(self, first_condition, second_condition):
        for (first_allele, second_allele) in zip(first_condition.genotype,
                                                 second_condition.genotype):
            if not self._is_wildcard(first_allele) and \
                    first_allele != second_allele:
                return False
        return True

    def map_genotype_to_phenotype(self, genotype):
        return tuple(genotype)

import abc

from piecewise.dtype import FloatAllele, IntegerAllele
from piecewise.environment import EnvironmentDtypes

from ..rule_repr import IRuleRepr


class IntervalRuleReprABC(IRuleRepr, metaclass=abc.ABCMeta):
    def __init__(self, situation_space, env_dtype):
        self._situation_space = situation_space
        self._wildcard_predicates = self._create_wildcard_predicates()
        self._allele_type = self._init_allele_type(env_dtype)

    def _create_wildcard_predicates(self):
        return tuple([
            self._create_wildcard_predicate_for_dim(dimension)
            for dimension in self._situation_space
        ])

    @abc.abstractmethod
    def _create_wildcard_predicate_for_dim(self, dimension):
        """Creates a wildcard predicate for the given dimension of the situation
        space.

        Dependent on the specific interval representation, hence is abstract
        and is up to subclasses to implement."""
        raise NotImplementedError

    def _init_allele_type(self, env_dtype):
        if env_dtype == EnvironmentDtypes.discrete:
            return IntegerAllele
        elif env_dtype == EnvironmentDtypes.continuous:
            return FloatAllele
        else:
            raise Exception

    def does_match(self, condition, situation):
        phenotype = self.genotype_to_phenotype(condition)
        for (interval_predicate, situation_elem) in zip(phenotype, situation):
            if not interval_predicate.lower() <= situation_elem <= \
                    interval_predicate.upper():
                return False
        return True

    @abc.abstractmethod
    def gen_covering_condition(self, situation):
        raise NotImplementedError

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        crossover_strat(first_condition, second_condition)

    @abc.abstractmethod
    def mutate_condition(self, condition, situation=None):
        raise NotImplementedError

    def _is_wildcard(self, interval_predicate, phenotype_idx):
        return interval_predicate.contains(
            self._wildcard_predicates[phenotype_idx])

    @abc.abstractmethod
    def genotype_to_phenotype(self, condition):
        raise NotImplementedError

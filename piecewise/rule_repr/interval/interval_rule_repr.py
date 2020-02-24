import abc

from ..rule_repr import IRuleRepr


class IntervalRuleReprABC(IRuleRepr, metaclass=abc.ABCMeta):
    def __init__(self, situation_space):
        self._situation_space = situation_space
        self._wildcards = self._create_wildcards()

    def _create_wildcards(self):
        return tuple([
            self._create_wildcard_for_dim(dimension)
            for dimension in self._situation_space
        ])

    @abc.abstractmethod
    def _create_wildcard_for_dim(self, dimension):
        """Creates a wildcard element for the given dimension of the situation
        space.

        Dependent on the specific interval representation, hence is abstract
        and is up to subclasses to implement."""
        raise NotImplementedError

    def does_match(self, condition, situation):
        for (condition_elem, situation_elem) in zip(condition.elems,
                                                    situation):
            if not condition_elem.lower() <= situation_elem <= \
                    condition_elem.upper():
                return False
        return True

    @abc.abstractmethod
    def gen_covering_condition(self, situation):
        raise NotImplementedError

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        crossover_strat(first_condition.alleles, second_condition.alleles)

    def mutate_condition(self, condition, situation=None):
        for condition_elem in condition.elems:
            condition_elem.mutate()

    def is_wildcard(self, condition_elem, elem_idx):
        return self._is_equiv_to_wildcard(condition_elem,
                                          self._wildcards[elem_idx])

    def num_wildcards(self, condition):
        num_wildcards = 0
        for (condition_elem, wildcard_elem) in zip(condition.elems,
                                                   self._wildcards):
            if self._is_equiv_to_wildcard(condition_elem, wildcard_elem):
                num_wildcards += 1
        return num_wildcards

    def _is_equiv_to_wildcard(self, condition_elem, wildcard_elem):
        return self._wildcard_is_sub_interval(wildcard_elem.interval(),
                                              condition_elem.interval())

    def _wildcard_is_sub_interval(self, wildcard_elem_interval,
                                  condition_elem_interval):
        return condition_elem_interval.lower <= wildcard_elem_interval.lower \
                and condition_elem_interval.upper >= \
                wildcard_elem_interval.upper

import abc
import random

from piecewise.dtype import Condition, IntegerAllele

from ..rule_repr import RuleRepr
from .elem.discrete_elem import DiscreteElem, DiscreteWildcardElem


class DiscreteRuleRepr(RuleRepr, metaclass=abc.ABCMeta):
    _WILDCARD_ELEM = DiscreteWildcardElem()

    def __init__(self, elem_value_set):
        self._elem_value_set = elem_value_set

    @property
    def elem_value_set(self):
        return self._elem_value_set

    def does_match(self, condition, situation):
        """DOES MATCH function from 'An Algorithmic Description of XCS'
        (Butz and Wilson, 2002)."""
        for (condition_elem, situation_elem) in zip(condition, situation):
            condition_elem_is_wildcard = self.is_wildcard(condition_elem)
            condition_elem_matches_situation = condition_elem == situation_elem
            if not condition_elem_is_wildcard and \
                    not condition_elem_matches_situation:
                return False
        return True

    def gen_covering_condition(self, situation, hyperparams):
        """First part (condition generation) of
        GENERATE COVERING CLASSIFIER function from
        'An Algorithmic Description of XCS' (Butz and Wilson, 2002).
        """
        condition = Condition()
        for situation_elem in situation:
            should_use_wildcard = random.random() < hyperparams["p_wildcard"]
            if should_use_wildcard:
                condition.append(self._make_wildcard_elem())
            else:
                condition.append(self._copy_situation_elem(situation_elem))
        return condition

    def mutate_condition(self, condition, hyperparams, situation):
        """First part (condition mutation) of APPLY MUTATION function from 'An
        Algorithmic Description of XCS' (Butz and Wilson, 2002)."""
        for elem_idx, (condition_elem,
                       situation_elem) in enumerate(zip(condition, situation)):

            should_mutate_elem = random.random() < hyperparams["mu"]
            if should_mutate_elem:
                if self.is_wildcard(condition_elem):
                    condition[elem_idx] = \
                        self._copy_situation_elem(situation_elem)
                else:
                    condition[elem_idx] = self._make_wildcard_elem()

    def is_wildcard(self, condition_elem, elem_idx=None):
        return condition_elem == self._WILDCARD_ELEM

    def num_wildcards(self, condition):
        return [
            self.is_wildcard(condition_elem) for condition_elem in condition
        ].count(True)

    def _make_wildcard_elem(self):
        return DiscreteWildcardElem()

    def _copy_situation_elem(self, situation_elem):
        return DiscreteElem(IntegerAllele(situation_elem))

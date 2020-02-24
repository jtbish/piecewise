from piecewise.dtype import DiscreteCondition, IntegerAllele
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng

from ..rule_repr import IRuleRepr
from .elem.discrete_elem import DiscreteElem, DiscreteWildcardElem


class DiscreteRuleRepr(IRuleRepr):
    """Rule representation that works with discrete (i.e. integer) inputs,
    storing a single integer in each element of a condition.

    Uses the DiscreteElem class to encapsulate condition elements, as well as
    the DiscreteWildcardElem as a sentinel class for representing wildcard
    elements."""
    _WILDCARD_ELEM = DiscreteWildcardElem()

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

    def gen_covering_condition(self, situation):
        """First part (condition generation) of
        GENERATE COVERING CLASSIFIER function from
        'An Algorithmic Description of XCS' (Butz and Wilson, 2002).
        """
        condition_elems = []
        for situation_elem in situation:
            should_use_wildcard = get_rng().rand() < get_hyperparam(
                "p_wildcard")
            if should_use_wildcard:
                condition_elems.append(self._make_wildcard_elem())
            else:
                condition_elems.append(
                    self._copy_situation_elem(situation_elem))
        return DiscreteCondition(condition_elems)

    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        crossover_strat(first_condition, second_condition)

    def mutate_condition(self, condition, situation):
        """First part (condition mutation) of APPLY MUTATION function from 'An
        Algorithmic Description of XCS' (Butz and Wilson, 2002)."""
        for elem_idx, (condition_elem,
                       situation_elem) in enumerate(zip(condition, situation)):

            should_mutate_elem = get_rng().rand() < get_hyperparam("mu")
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

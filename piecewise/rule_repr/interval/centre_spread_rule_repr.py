import random

from piecewise.dtype import Condition, FloatAllele
from .interval_rule_repr import IntervalRuleRepr
from .elem.centre_spread_elem import CentreSpreadElem


class CentreSpreadRuleRepr(IntervalRuleRepr):
    def __init__(self, situation_space):
        super().__init__(situation_space)

    def _create_wildcard_for_dim(self, dimension):
        centre = (dimension.upper + dimension.lower) / 2
        spread = dimension.upper - centre
        return self._make_elem(centre, spread)

    def gen_covering_condition(self, situation, hyperparams):
        """Implementation of covering for XCSR as described in 'Get Real! XCS
        With Continuous-Valued Inputs' (Wilson, 2000)."""
        condition = Condition()
        for situation_elem in situation:
            centre = situation_elem
            spread = random.uniform(0, hyperparams["s_nought"])
            condition.append(self._make_elem(centre, spread))

        return condition

    def _make_elem(self, centre, spread):
        return CentreSpreadElem(FloatAllele(centre), FloatAllele(spread))

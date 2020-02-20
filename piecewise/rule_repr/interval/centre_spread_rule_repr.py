from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng
from piecewise.dtype import Condition, FloatAllele

from .elem.centre_spread_elem import CentreSpreadElem
from .interval_rule_repr import IntervalRuleReprABC


def make_centre_spread_rule_repr(env):
    return CentreSpreadRuleRepr(situation_space=env.obs_space)


class CentreSpreadRuleRepr(IntervalRuleReprABC):
    """Rule representation that works with (centre, spread) tuples."""
    def __init__(self, situation_space):
        super().__init__(situation_space)

    def _create_wildcard_for_dim(self, dimension):
        centre = (dimension.upper + dimension.lower) / 2
        spread = dimension.upper - centre
        return self._make_elem(centre, spread)

    def gen_covering_condition(self, situation):
        """Implementation of covering for XCSR as described in 'Get Real! XCS
        With Continuous-Valued Inputs' (Wilson, 2000)."""
        condition = Condition()
        for situation_elem in situation:
            centre = situation_elem
            spread = get_rng().uniform(0, get_hyperparam("s_nought"))
            condition.append(self._make_elem(centre, spread))

        return condition

    def _make_elem(self, centre, spread):
        return CentreSpreadElem(FloatAllele(centre), FloatAllele(spread))

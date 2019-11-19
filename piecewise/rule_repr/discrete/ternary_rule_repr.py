from .discrete_rule_repr import DiscreteRuleRepr


class TernaryRuleRepr(DiscreteRuleRepr):
    def __init__(self):
        elem_value_set = {0, 1}
        super().__init__(elem_value_set)

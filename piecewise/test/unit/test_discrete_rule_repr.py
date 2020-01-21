import pytest

from piecewise.rule_repr import DiscreteRuleRepr
from piecewise.rule_repr.discrete.elem.discrete_elem import \
    DiscreteWildcardElem


@pytest.fixture
def discrete_rule_repr():
    return DiscreteRuleRepr()


class TestDiscreteRuleRepr:
    def test_does_match_pos_case_wildcard(self, discrete_rule_repr):
        condition = [DiscreteWildcardElem()]
        situation = [0]
        assert discrete_rule_repr.does_match(condition, situation)

    def test_does_match_pos_case_non_wildcard(self, discrete_rule_repr):
        condition = [0]
        situation = [0]
        assert discrete_rule_repr.does_match(condition, situation)

    def test_does_match_neg_case(self, discrete_rule_repr):
        condition = [1]
        situation = [0]
        assert not discrete_rule_repr.does_match(condition, situation)

    def test_gen_covering_condition_force_create_wildcard(
            self, discrete_rule_repr):
        situation = [0]
        hyperparams = {"p_wildcard": 1.0}
        condition = discrete_rule_repr.gen_covering_condition(
            situation, hyperparams)
        assert len(condition) == 1
        assert condition[0] == DiscreteWildcardElem()

    def test_gen_covering_condition_force_copy_situation(
            self, discrete_rule_repr):
        situation = [0]
        hyperparams = {"p_wildcard": 0.0}
        condition = discrete_rule_repr.gen_covering_condition(
            situation, hyperparams)
        assert len(condition) == 1
        assert condition[0] == 0

    def test_num_wildcards_zero(self, discrete_rule_repr):
        condition = [0, 0]
        assert discrete_rule_repr.num_wildcards(condition) == 0

    def test_num_wildcards_some(self, discrete_rule_repr):
        condition = [0, DiscreteWildcardElem()]
        assert discrete_rule_repr.num_wildcards(condition) == 1

    def test_num_wildcards_all(self, discrete_rule_repr):
        condition = [DiscreteWildcardElem()] * 2
        assert discrete_rule_repr.num_wildcards(condition) == 2

    def test_mutate_condition_force_mutate_to_wildcard(self,
                                                       discrete_rule_repr):
        condition = [0]
        # situation does not matter in this context
        situation = [1]
        hyperparams = {"mu": 1.0}
        discrete_rule_repr.mutate_condition(condition, hyperparams, situation)
        assert len(condition) == 1
        assert condition[0] == DiscreteWildcardElem()

    def test_mutate_condition_force_mutate_to_situation(
            self, discrete_rule_repr):
        condition = [DiscreteWildcardElem()]
        situation = [0]
        hyperparams = {"mu": 1.0}
        discrete_rule_repr.mutate_condition(condition, hyperparams, situation)
        assert len(condition) == 1
        assert condition[0] == 0

    def test_mutate_condition_force_stay_non_wildcard(self,
                                                      discrete_rule_repr):
        condition = [0]
        situation = [1]
        hyperparams = {"mu": 0.0}
        discrete_rule_repr.mutate_condition(condition, hyperparams, situation)
        assert len(condition) == 1
        assert condition[0] == 0

    def test_mutate_condition_force_stay_wildcard(self, discrete_rule_repr):
        condition = [DiscreteWildcardElem()]
        situation = [1]
        hyperparams = {"mu": 0.0}
        discrete_rule_repr.mutate_condition(condition, hyperparams, situation)
        assert len(condition) == 1
        assert condition[0] == DiscreteWildcardElem()

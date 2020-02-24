import pytest

from piecewise.dtype import DataSpaceBuilder, Dimension, IntervalCondition
from piecewise.rule_repr.interval.centre_spread_rule_repr import \
    CentreSpreadRuleRepr
from piecewise.rule_repr.interval.elem.centre_spread_elem import \
    CentreSpreadElem
from piecewise.rule_repr.interval.elem.interval_elem import Interval
from piecewise.lcs.hyperparams import register_hyperparams

TESTING_DIMS = [(-1.0, 1.0), (-1.0, 0.0), (0.0, 1.0), (-10.0, 10.0),
                (-50.0, 10.0), (-10.0, 50.0), (-100.0, 0.0), (0.0, 100.0),
                (-100.0, 100.0), (-33.3, 33.3)]
TESTING_DIMS = [Dimension(lower, upper) for (lower, upper) in TESTING_DIMS]


@pytest.fixture
def make_single_dim_situation_space():
    def _make_single_dim_situation_space(dim):
        builder = DataSpaceBuilder()
        builder.add_dim(dim)
        return builder.create_space()

    return _make_single_dim_situation_space


@pytest.fixture
def mock_situation_space(mocker):
    return mocker.MagicMock()


@pytest.fixture
def make_centre_spread_rule_repr():
    def _make_centre_spread_rule_repr(situation_space):
        return CentreSpreadRuleRepr(situation_space)

    return _make_centre_spread_rule_repr


@pytest.fixture
def make_mock_condition_elem(mocker):
    def _make_mock_condition_elem(lower, upper):
        elem = mocker.MagicMock()
        elem.lower.return_value = lower
        elem.upper.return_value = upper
        elem.interval.return_value = Interval(lower, upper)
        elem.blah.return_value = 26
        return elem

    return _make_mock_condition_elem


@pytest.fixture
def make_mock_single_elem_condition(make_mock_condition_elem):
    def _make_mock_single_elem_condition(lower, upper):
        elem = make_mock_condition_elem(lower, upper)
        return IntervalCondition.from_elem_args(elem)

    return _make_mock_single_elem_condition


@pytest.fixture(params=TESTING_DIMS)
def setup_is_wildcard_test(request, make_single_dim_situation_space,
                           make_centre_spread_rule_repr):
    dim = request.param
    situation_space = make_single_dim_situation_space(dim)
    rule_repr = make_centre_spread_rule_repr(situation_space)
    return dim, rule_repr


class TestCentreSpreadRuleRepr:
    _INTERVAL_DIFF = 1.0

    def _first_elem_is_wildcard(self, rule_repr, condition_elem):
        return rule_repr.is_wildcard(condition_elem, 0)

    def test_does_match_pos_case_contained(self, mocker, mock_situation_space,
                                           make_centre_spread_rule_repr,
                                           make_mock_single_elem_condition):
        rule_repr = make_centre_spread_rule_repr(mock_situation_space)
        condition = make_mock_single_elem_condition(lower=-1.0, upper=1.0)
        situation = [0.0]
        assert rule_repr.does_match(condition, situation)

    def test_does_match_pos_eq_lower(self, mocker, mock_situation_space,
                                     make_centre_spread_rule_repr,
                                     make_mock_single_elem_condition):
        rule_repr = make_centre_spread_rule_repr(mock_situation_space)
        lower = -1.0
        condition = make_mock_single_elem_condition(lower=lower, upper=1.0)
        situation = [lower]
        assert rule_repr.does_match(condition, situation)

    def test_does_match_pos_eq_upper(self, mocker, mock_situation_space,
                                     make_centre_spread_rule_repr,
                                     make_mock_single_elem_condition):
        rule_repr = make_centre_spread_rule_repr(mock_situation_space)
        upper = 1.0
        condition = make_mock_single_elem_condition(lower=-upper, upper=upper)
        situation = [upper]
        assert rule_repr.does_match(condition, situation)

    def test_does_match_neg_case_lt_lower(self, mocker, mock_situation_space,
                                          make_centre_spread_rule_repr,
                                          make_mock_single_elem_condition):
        rule_repr = make_centre_spread_rule_repr(mock_situation_space)
        lower = -1.0
        condition = make_mock_single_elem_condition(lower=lower, upper=1.0)
        situation = [lower * 2]
        assert not rule_repr.does_match(condition, situation)

    def test_does_match_neg_case_gt_upper(self, mocker, mock_situation_space,
                                          make_centre_spread_rule_repr,
                                          make_mock_single_elem_condition):
        rule_repr = make_centre_spread_rule_repr(mock_situation_space)
        upper = 1.0
        condition = make_mock_single_elem_condition(lower=-1.0, upper=upper)
        situation = [upper * 2]
        assert not rule_repr.does_match(condition, situation)

    def test_gen_covering_condition(self, mock_situation_space,
                                    make_centre_spread_rule_repr):
        rule_repr = make_centre_spread_rule_repr(mock_situation_space)
        register_hyperparams({"s_nought": 1.0})
        situation_elem = 0.0
        situation = [situation_elem]
        condition = rule_repr.gen_covering_condition(situation)
        assert len(condition) == 1
        elem = condition[0]
        assert type(elem) == CentreSpreadElem
        assert elem._centre_allele == situation_elem

    def test_is_wildcard_pos_case_same_interval(self, setup_is_wildcard_test,
                                                make_mock_condition_elem):
        dim, rule_repr = setup_is_wildcard_test
        condition_elem = \
            make_mock_condition_elem(lower=dim.lower,
                                     upper=dim.upper)
        assert self._first_elem_is_wildcard(rule_repr, condition_elem)

    def test_is_wildcard_pos_case_super_interval(self, setup_is_wildcard_test,
                                                 make_mock_condition_elem):
        dim, rule_repr = setup_is_wildcard_test
        condition_elem = \
            make_mock_condition_elem(lower=(dim.lower - self._INTERVAL_DIFF),
                                     upper=(dim.upper + self._INTERVAL_DIFF))
        assert self._first_elem_is_wildcard(rule_repr, condition_elem)

    def test_is_wildcard_neg_case_sub_interval(self, setup_is_wildcard_test,
                                               make_mock_condition_elem):
        dim, rule_repr = setup_is_wildcard_test
        condition_elem = \
            make_mock_condition_elem(lower=(dim.lower + self._INTERVAL_DIFF),
                                     upper=(dim.upper - self._INTERVAL_DIFF))
        assert not self._first_elem_is_wildcard(rule_repr, condition_elem)

    def test_is_wildcard_neg_case_lower_too_high(self, setup_is_wildcard_test,
                                                 make_mock_condition_elem):
        dim, rule_repr = setup_is_wildcard_test
        condition_elem = \
            make_mock_condition_elem(lower=(dim.lower + self._INTERVAL_DIFF),
                                     upper=(dim.upper))
        assert not self._first_elem_is_wildcard(rule_repr, condition_elem)

    def test_is_wildcard_neg_case_upper_too_low(self, setup_is_wildcard_test,
                                                make_mock_condition_elem):
        dim, rule_repr = setup_is_wildcard_test
        condition_elem = \
            make_mock_condition_elem(lower=(dim.lower),
                                     upper=(dim.upper - self._INTERVAL_DIFF))
        assert not self._first_elem_is_wildcard(rule_repr, condition_elem)

    def test_num_wildcards_zero(self, make_single_dim_situation_space,
                                make_centre_spread_rule_repr,
                                make_mock_single_elem_condition):
        lower = -1.0
        upper = 1.0
        dim = Dimension(lower, upper)
        situation_space = make_single_dim_situation_space(dim)
        rule_repr = make_centre_spread_rule_repr(situation_space)
        condition = make_mock_single_elem_condition(
            lower=(lower + self._INTERVAL_DIFF),
            upper=(upper - self._INTERVAL_DIFF))
        assert rule_repr.num_wildcards(condition) == 0

    def test_num_wildcards_one(self, make_single_dim_situation_space,
                               make_centre_spread_rule_repr,
                               make_mock_single_elem_condition):
        lower = -1.0
        upper = 1.0
        dim = Dimension(lower, upper)
        situation_space = make_single_dim_situation_space(dim)
        rule_repr = make_centre_spread_rule_repr(situation_space)
        condition = make_mock_single_elem_condition(lower, upper)
        assert rule_repr.num_wildcards(condition) == 1

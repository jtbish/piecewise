import pytest

from piecewise.dtype import Rule


@pytest.fixture
def make_mock_condition(mocker):
    def _make_mock_condition():
        return mocker.MagicMock()

    return _make_mock_condition


@pytest.fixture
def make_mock_action(mocker):
    def _make_mock_action():
        return mocker.MagicMock()

    return _make_mock_action


@pytest.fixture
def mock_condition(make_mock_condition):
    return make_mock_condition()


@pytest.fixture
def mock_action(make_mock_action):
    return make_mock_action()


class TestRule:
    def test_eq_pos_case(self, mock_condition, mock_action):
        rule = Rule(mock_condition, mock_action)
        same_rule = Rule(mock_condition, mock_action)
        assert rule == same_rule

    def test_ne_pos_case_diff_condition(self, make_mock_condition,
                                        mock_action):
        condition = make_mock_condition()
        rule = Rule(condition, mock_action)
        diff_condition = make_mock_condition()
        diff_condition_rule = Rule(diff_condition, mock_action)
        assert rule != diff_condition_rule

    def test_ne_pos_case_diff_action(self, mock_condition, make_mock_action):
        action = make_mock_action()
        rule = Rule(mock_condition, action)
        diff_action = make_mock_action()
        diff_action_rule = Rule(mock_condition, diff_action)
        assert rule != diff_action_rule

    def test_ne_pos_case_both_diff(self, make_mock_condition,
                                   make_mock_action):
        condition = make_mock_condition()
        action = make_mock_action()
        rule = Rule(condition, action)
        diff_condition = make_mock_condition()
        diff_action = make_mock_action()
        both_diff_rule = Rule(diff_condition, diff_action)
        assert rule != both_diff_rule

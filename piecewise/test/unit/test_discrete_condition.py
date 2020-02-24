import pytest

from piecewise.dtype import DiscreteCondition


@pytest.fixture
def empty_condition():
    return DiscreteCondition()


@pytest.fixture
def one_elem_condition(mock_elem):
    return DiscreteCondition.from_elem_args(mock_elem)


class TestDiscreteCondition:
    def test_append(self, mock_elem):
        condition = DiscreteCondition.from_elem_args(mock_elem)
        assert condition._elems[-1] == mock_elem

    def test_eq_pos_case(self, make_mock_elem):
        elem = make_mock_elem()
        condition = DiscreteCondition.from_elem_args(elem)
        same_condition = DiscreteCondition.from_elem_args(elem)
        assert condition == same_condition

    def test_ne_pos_case(self, make_mock_elem):
        elem = make_mock_elem()
        condition = DiscreteCondition.from_elem_args(elem)
        diff_elem = make_mock_elem()
        diff_condition = DiscreteCondition.from_elem_args(diff_elem)
        assert condition != diff_condition

    def test_setitem_non_neg_index(self, one_elem_condition, make_mock_elem):
        new_elem = make_mock_elem()
        one_elem_condition[0] = new_elem
        assert one_elem_condition[0] == new_elem

    def test_setitem_neg_index(self, one_elem_condition, make_mock_elem):
        new_elem = make_mock_elem()
        one_elem_condition[-1] = new_elem
        assert one_elem_condition[-1] == new_elem

    def test_setitem_bad_index(self, one_elem_condition, make_mock_elem):
        with pytest.raises(IndexError):
            one_elem_condition[1] = make_mock_elem()

    def test_getitem_non_neg_index(self, mock_elem):
        condition = DiscreteCondition.from_elem_args(mock_elem)
        assert condition[0] == mock_elem

    def test_getitem_neg_index(self, mock_elem):
        condition = DiscreteCondition.from_elem_args(mock_elem)
        assert condition[-1] == mock_elem

    def test_getitem_bad_index(self, one_elem_condition):
        with pytest.raises(IndexError):
            one_elem_condition[2]

    def test_len(self, one_elem_condition):
        assert len(one_elem_condition) == 1

    def test_iter(self, mock_elem):
        condition = DiscreteCondition.from_elem_args(mock_elem)
        elems = []
        for elem in condition:
            elems.append(elem)
        assert elems == [mock_elem]

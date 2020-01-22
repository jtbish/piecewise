import math

from piecewise.component.prediction import PredictionArray


class TestPredictionArray:
    def test_getitem_transforms_null_into_zero(self):
        pred_arr = PredictionArray({0})
        assert pred_arr[0] == 0.0

    def test_setitem_on_unset_action_then_getitem(self):
        pred_arr = PredictionArray({0})
        pred_arr[0] = math.pi
        assert pred_arr[0] == math.pi

    def test_overwrite_prediction_on_already_set_action(self):
        pred_arr = PredictionArray({0})
        pred_arr[0] = math.pi
        pred_arr[0] = math.e
        assert pred_arr[0] == math.e

    def test_possible_actions_set_empty(self):
        pred_arr = PredictionArray({0, 1})
        assert pred_arr.possible_actions_set() == set()

    def test_possible_actions_set_single_member(self):
        pred_arr = PredictionArray({0, 1})
        pred_arr[0] = math.pi
        assert pred_arr.possible_actions_set() == {0}

    def test_possible_sub_array_empty(self):
        pred_arr = PredictionArray({0, 1})
        assert pred_arr.possible_sub_array() == {}

    def test_possible_sub_array_single_member(self):
        pred_arr = PredictionArray({0, 1})
        pred_arr[0] = math.pi
        assert pred_arr.possible_sub_array() == {0: math.pi}

    def test_getitem_on_unset_action_does_not_alter_possible_actions(self):
        pred_arr = PredictionArray({0})
        pred_arr[0]
        assert pred_arr.possible_actions_set() == set()

    def test_getitem_on_unset_action_does_not_alter_possible_sub_array(self):
        pred_arr = PredictionArray({0})
        pred_arr[0]
        assert pred_arr.possible_sub_array() == {}

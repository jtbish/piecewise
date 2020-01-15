import math

from piecewise.component.common.prediction import PredictionArray


class TestPredictionArray:
    def test_getitem_transforms_null_into_zero(self):
        pred_arr = PredictionArray({0})
        assert pred_arr[0] == 0.0

    def test_getitem_on_non_null_action(self):
        pred_arr = PredictionArray({0})
        pred_arr[0] = math.pi
        assert pred_arr[0] == math.pi

    def test_setitem_on_unset_action(self):
        pred_arr = PredictionArray({0})
        pred_arr[0] = math.pi
        assert pred_arr[0] == math.pi

    def test_setitem_on_already_set_action(self):
        pred_arr = PredictionArray({0})
        pred_arr[0] = math.pi
        pred_arr[0] = math.e
        assert pred_arr[0] == math.e

    def test_random_action_single_option(self):
        env_action_set = {0, 1}
        pred_arr = PredictionArray(env_action_set)
        pred_arr[0] = math.pi
        # 1 is null
        assert pred_arr.random_action() == 0

    def test_random_action_multiple_options(self):
        env_action_set = {0, 1, 2}
        pred_arr = PredictionArray(env_action_set)
        pred_arr[0] = math.pi
        pred_arr[1] = math.pi
        # 2 is null
        random_action = pred_arr.random_action()
        assert random_action == 0 or random_action == 1

    def test_greedy_action_non_ambiguous(self):
        pass

    def test_greedy_action_non_ambiguous_with_null_prediction(self):
        pass

    def test_greedy_action_ambiguous(self):
        pass

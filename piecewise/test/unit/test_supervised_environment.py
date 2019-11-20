import numpy as np
import pytest

from piecewise.environment import DiscreteMultiplexer, EnvironmentStepTypes
from piecewise.error.environment_error import OutOfDataError


class TestSupervisedEnvironmentViaDiscreteMultiplexer:
    _DUMMY_ACTION = 0

    def _setup_short_epoch(self):
        num_address_bits = 1
        total_bits = num_address_bits + 2**num_address_bits
        num_data_points = 2**total_bits
        mux = DiscreteMultiplexer(num_address_bits=num_address_bits,
                                  shuffle_dataset=False)
        return mux, num_data_points

    def test_obs_space_integrity(self):
        num_address_bits = 1
        mux = DiscreteMultiplexer(num_address_bits=1)
        total_feature_dims = num_address_bits + 2**num_address_bits
        assert len(mux.obs_space) == total_feature_dims
        for dim in mux.obs_space:
            assert dim.lower == 0
            assert dim.upper == 1

    def test_action_set_integrity(self):
        mux = DiscreteMultiplexer()
        assert mux.action_set == {0, 1}

    def test_step_type(self):
        mux = DiscreteMultiplexer()
        assert mux.step_type == EnvironmentStepTypes.single_step

    def test_observe_order_no_shuffle(self):
        mux = DiscreteMultiplexer(num_address_bits=1, shuffle_dataset=False)
        expected_obs_seq_iter = \
            iter([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                  [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        while not mux.is_terminal():
            obs = mux.observe()
            assert np.array_equal(obs, next(expected_obs_seq_iter))
            mux.act(self._DUMMY_ACTION)

    def test_act_all_correct(self):
        mux = DiscreteMultiplexer(num_address_bits=1, shuffle_dataset=False)
        correct_actions_iter = iter([0, 0, 1, 1, 0, 1, 0, 1])

        while not mux.is_terminal():
            response = mux.act(next(correct_actions_iter))
            assert response.was_correct_action

    def test_act_all_incorrect(self):
        mux = DiscreteMultiplexer(num_address_bits=1, shuffle_dataset=False)
        incorrect_actions_iter = iter([1, 1, 0, 0, 1, 0, 1, 0])

        while not mux.is_terminal():
            response = mux.act(next(incorrect_actions_iter))
            assert not response.was_correct_action

    def test_act_changes_next_obs(self):
        mux, num_data_points = self._setup_short_epoch()
        last_obs = None

        for _ in range(num_data_points):
            obs = mux.observe()
            if last_obs is not None:
                assert not np.array_equal(last_obs, obs)
            mux.act(self._DUMMY_ACTION)
            last_obs = obs

    def test_same_obs_on_repeated_observe(self):
        mux, num_data_points = self._setup_short_epoch()
        last_obs = None

        for _ in range(num_data_points):
            obs = mux.observe()
            if last_obs is not None:
                assert np.array_equal(last_obs, obs)
            last_obs = obs

    def test_is_terminal_act_only_epoch(self):
        mux, num_data_points = self._setup_short_epoch()
        for _ in range(num_data_points):
            mux.act(self._DUMMY_ACTION)
        assert mux.is_terminal()

    def test_is_not_terminal_observe_only_epoch(self):
        mux, num_data_points = self._setup_short_epoch()
        for _ in range(num_data_points):
            mux.observe()
        assert not mux.is_terminal()

    def test_is_terminal_act_and_observe_epoch(self):
        mux, num_data_points = self._setup_short_epoch()
        for _ in range(num_data_points):
            mux.observe()
            mux.act(self._DUMMY_ACTION)
        assert mux.is_terminal()

    def test_out_of_data_on_extra_act(self):
        mux, num_data_points = self._setup_short_epoch()
        for _ in range(num_data_points):
            mux.observe()
            mux.act(self._DUMMY_ACTION)
        with pytest.raises(OutOfDataError):
            mux.act(self._DUMMY_ACTION)

    def test_out_of_data_on_extra_observe(self):
        mux, num_data_points = self._setup_short_epoch()
        for _ in range(num_data_points):
            mux.observe()
            mux.act(self._DUMMY_ACTION)
        with pytest.raises(OutOfDataError):
            mux.observe()

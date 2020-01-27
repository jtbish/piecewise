import numpy as np
import pytest

from piecewise.environment import EnvironmentStepTypes, make_discrete_mux_env
from piecewise.environment.supervised.multiplexer.multiplexer_util import \
    calc_total_bits
from piecewise.error.environment_error import OutOfDataError


class TestClassificationEnvironmentViaDiscreteMultiplexer:
    _DUMMY_ACTION = 0

    def _setup_short_epoch(self):
        num_address_bits = 1
        total_bits = calc_total_bits(num_address_bits)
        num_data_points = 2**total_bits
        mux = make_discrete_mux_env(num_address_bits=num_address_bits,
                                    shuffle_dataset=False)
        return mux, num_data_points

    def test_step_type(self):
        mux = make_discrete_mux_env()
        assert mux.step_type == EnvironmentStepTypes.single_step

    def test_observe_order_no_shuffle(self):
        mux = make_discrete_mux_env(num_address_bits=1, shuffle_dataset=False)
        expected_obs_seq_iter = \
            iter([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                  [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        while not mux.is_terminal():
            obs = mux.observe()
            assert np.array_equal(obs, next(expected_obs_seq_iter))
            mux.act(self._DUMMY_ACTION)

    def test_act_all_correct(self):
        mux = make_discrete_mux_env(num_address_bits=1, shuffle_dataset=False)
        correct_actions_iter = iter([0, 0, 1, 1, 0, 1, 0, 1])

        while not mux.is_terminal():
            response = mux.act(next(correct_actions_iter))
            assert response.was_correct_action

    def test_act_all_incorrect(self):
        mux = make_discrete_mux_env(num_address_bits=1, shuffle_dataset=False)
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

    def test_reset_with_two_epochs_no_shuffle(self):
        mux = make_discrete_mux_env(num_address_bits=1,
                                    shuffle_dataset=False)

        first_epoch_obs_seq = []
        first_epoch_reward_seq = []
        while not mux.is_terminal():
            first_epoch_obs_seq.append(mux.observe())
            response = mux.act(self._DUMMY_ACTION)
            first_epoch_reward_seq.append(response.reward)

        mux.reset()
        second_epoch_obs_seq = []
        second_epoch_reward_seq = []
        while not mux.is_terminal():
            second_epoch_obs_seq.append(mux.observe())
            response = mux.act(self._DUMMY_ACTION)
            second_epoch_reward_seq.append(response.reward)

        assert np.array_equal(first_epoch_obs_seq, second_epoch_obs_seq)
        assert np.array_equal(first_epoch_reward_seq, second_epoch_reward_seq)

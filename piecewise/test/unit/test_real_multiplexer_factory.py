import numpy as np
import pytest

from piecewise.environment import make_real_mux_env
from piecewise.environment.supervised.multiplexer.multiplexer_builders import \
    RealMultiplexerBuilder
from piecewise.environment.supervised.multiplexer.multiplexer_util import \
    calc_total_bits
from piecewise.error.environment_error import InvalidSpecError

THRESHOLD_MIN = RealMultiplexerBuilder.THRESHOLD_MIN
THRESHOLD_MAX = RealMultiplexerBuilder.THRESHOLD_MAX


class TestRealMultiplexerFactory:
    def test_bad_num_address_bits_zero(self):
        with pytest.raises(InvalidSpecError):
            make_real_mux_env(num_address_bits=0, thresholds=[])

    def test_bad_num_address_bits_negative(self):
        with pytest.raises(InvalidSpecError):
            make_real_mux_env(num_address_bits=-1, thresholds=[])

    def test_bad_given_thresholds_incorrect_len_1d_array(self):
        num_address_bits = 1
        total_bits = calc_total_bits(num_address_bits)
        bad_len_thresholds = [0.5] * (total_bits - 1)
        with pytest.raises(InvalidSpecError):
            make_real_mux_env(num_address_bits=num_address_bits,
                              thresholds=bad_len_thresholds)

    def test_bad_given_thresholds_incorrect_len_2d_array(self):
        num_address_bits = 1
        total_bits = calc_total_bits(num_address_bits)
        bad_len_thresholds = [[0.5] * total_bits, [0.5] * total_bits]
        with pytest.raises(InvalidSpecError):
            make_real_mux_env(num_address_bits=num_address_bits,
                              thresholds=bad_len_thresholds)

    def test_bad_given_thresholds_not_float(self):
        num_address_bits = 1
        total_bits = calc_total_bits(num_address_bits)
        non_float_thresholds = [int(THRESHOLD_MIN)] * total_bits
        with pytest.raises(InvalidSpecError):
            make_real_mux_env(num_address_bits=num_address_bits,
                              thresholds=non_float_thresholds)

    def test_bad_given_thresholds_not_in_valid_range(self):
        num_address_bits = 1
        total_bits = calc_total_bits(num_address_bits)
        bad_range_thresholds = [THRESHOLD_MAX * 2] * total_bits
        with pytest.raises(InvalidSpecError):
            make_real_mux_env(num_address_bits=num_address_bits,
                              thresholds=bad_range_thresholds)

    def test_data_values(self):
        num_address_bits = 1
        total_bits = calc_total_bits(num_address_bits)
        thresholds = [0.5] * total_bits
        mux = make_real_mux_env(num_address_bits=num_address_bits,
                                thresholds=thresholds)
        assert np.all(0.0 <= mux.data) and np.all(mux.data < 1.0)

    def test_data_dtype(self):
        num_address_bits = 1
        total_bits = calc_total_bits(num_address_bits)
        thresholds = [0.5] * total_bits
        mux = make_real_mux_env(num_address_bits=num_address_bits,
                                thresholds=thresholds)
        for elem in mux.data.flatten():
            assert isinstance(elem, np.floating)

    def test_labels_values(self):
        num_address_bits = 1
        total_bits = calc_total_bits(num_address_bits)
        thresholds = [0.5] * total_bits
        mux = make_real_mux_env(num_address_bits=num_address_bits,
                                thresholds=thresholds)
        assert np.all((mux.labels == 0) | (mux.labels == 1))

    def test_labels_dtype(self):
        num_address_bits = 1
        total_bits = calc_total_bits(num_address_bits)
        thresholds = [0.5] * total_bits
        mux = make_real_mux_env(num_address_bits=num_address_bits,
                                thresholds=thresholds)
        for elem in mux.labels.flatten():
            assert isinstance(elem, np.integer)

    def test_obs_space_integrity(self):
        num_address_bits = 1
        total_feature_dims = calc_total_bits(num_address_bits)
        thresholds = [0.5] * total_feature_dims
        mux = make_real_mux_env(thresholds=thresholds, num_address_bits=1)
        assert len(mux.obs_space) == total_feature_dims
        for dim in mux.obs_space:
            assert dim.lower == 0.0
            assert dim.upper == 1.0

    def test_action_set_integrity(self):
        num_address_bits = 1
        total_bits = calc_total_bits(num_address_bits)
        thresholds = [0.5] * total_bits
        mux = make_real_mux_env(thresholds=thresholds, num_address_bits=1)
        assert mux.action_set == {0, 1}

import numpy as np
import pytest

from piecewise.environment import RealMultiplexer
from piecewise.environment.supervised.multiplexer.abstract_multiplexer import \
    calc_total_bits
from piecewise.environment.supervised.multiplexer.real_multiplexer import (
    THRESHOLD_MAX, THRESHOLD_MIN)
from piecewise.error.environment_error import InvalidSpecError


class TestRealMultiplexer:
    def test_thresholds_values_no_given_thresholds(self):
        mux = RealMultiplexer()
        assert np.all(0.0 <= mux.thresholds) and np.all(mux.thresholds < 1.0)

    def test_thresholds_dtype_no_given_thresholds(self):
        mux = RealMultiplexer()
        for elem in mux.thresholds.flatten():
            assert isinstance(elem, np.floating)

    def test_bad_given_thresholds_incorrect_len(self):
        num_address_bits = 1
        total_bits = calc_total_bits(num_address_bits)
        bad_len_thresholds = [0.5] * (total_bits - 1)
        with pytest.raises(InvalidSpecError):
            RealMultiplexer(num_address_bits=num_address_bits,
                            thresholds=bad_len_thresholds)

    def test_bad_given_thresholds_not_float(self):
        num_address_bits = 1
        total_bits = calc_total_bits(num_address_bits)
        non_float_thresholds = [int(THRESHOLD_MIN)] * total_bits
        with pytest.raises(InvalidSpecError):
            RealMultiplexer(num_address_bits=num_address_bits,
                            thresholds=non_float_thresholds)

    def test_bad_given_thresholds_not_in_valid_range(self):
        num_address_bits = 1
        total_bits = calc_total_bits(num_address_bits)
        bad_range_thresholds = [THRESHOLD_MAX * 2] * total_bits
        with pytest.raises(InvalidSpecError):
            RealMultiplexer(num_address_bits=num_address_bits,
                            thresholds=bad_range_thresholds)

    def test_data_values(self):
        mux = RealMultiplexer()
        assert np.all(0.0 <= mux.data) and np.all(mux.data < 1.0)

    def test_data_dtype(self):
        mux = RealMultiplexer()
        for elem in mux.data.flatten():
            assert isinstance(elem, np.floating)

    def test_labels_values(self):
        mux = RealMultiplexer()
        assert np.all((mux.labels == 0) | (mux.labels == 1))

    def test_labels_dtype(self):
        mux = RealMultiplexer()
        for elem in mux.labels.flatten():
            assert isinstance(elem, np.integer)

import numpy as np
import pytest

from piecewise.environment import make_discrete_mux_env
from piecewise.environment.supervised.multiplexer.multiplexer_util import \
    calc_total_bits
from piecewise.error.environment_error import InvalidSpecError


class TestDiscreteMultiplexerFactory:
    def test_bad_num_address_bits_zero(self):
        with pytest.raises(InvalidSpecError):
            make_discrete_mux_env(num_address_bits=0)

    def test_bad_num_address_bits_negative(self):
        with pytest.raises(InvalidSpecError):
            make_discrete_mux_env(num_address_bits=-1)

    def test_data_values(self):
        mux = make_discrete_mux_env(num_address_bits=1)
        expected_data = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
                         [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        assert np.array_equal(mux.data, expected_data)

    def test_data_dtype(self):
        mux = make_discrete_mux_env()
        for elem in mux.data.flatten():
            assert isinstance(elem, np.integer)

    def test_labels_values(self):
        mux = make_discrete_mux_env(num_address_bits=1)
        expected_labels = [0, 0, 1, 1, 0, 1, 0, 1]
        assert np.array_equal(mux.labels, expected_labels)

    def test_labels_dtype(self):
        mux = make_discrete_mux_env()
        for elem in mux.labels.flatten():
            assert isinstance(elem, np.integer)

    def test_obs_space_integrity(self):
        num_address_bits = 1
        mux = make_discrete_mux_env(num_address_bits=1)
        total_feature_dims = calc_total_bits(num_address_bits)
        assert len(mux.obs_space) == total_feature_dims
        for dim in mux.obs_space:
            assert dim.lower == 0
            assert dim.upper == 1

    def test_action_set_integrity(self):
        mux = make_discrete_mux_env()
        assert mux.action_set == {0, 1}

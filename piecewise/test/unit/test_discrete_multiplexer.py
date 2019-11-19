import numpy as np

from piecewise.environment import DiscreteMultiplexer


class TestDiscreteMultiplexer:
    def test_data_values(self):
        mux = DiscreteMultiplexer(num_address_bits=1)
        expected_data = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
                         [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        assert np.array_equal(mux.data, expected_data)

    def test_data_dtype(self):
        mux = DiscreteMultiplexer()
        for elem in mux.data.flatten():
            assert isinstance(elem, np.integer)

    def test_labels_values(self):
        mux = DiscreteMultiplexer(num_address_bits=1)
        expected_labels = [0, 0, 1, 1, 0, 1, 0, 1]
        assert np.array_equal(mux.labels, expected_labels)

    def test_labels_dtype(self):
        mux = DiscreteMultiplexer()
        for elem in mux.labels.flatten():
            assert isinstance(elem, np.integer)

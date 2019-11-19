import itertools

from .abstract_multiplexer import AbstractMultiplexer


class DiscreteMultiplexer(AbstractMultiplexer):
    def __init__(self, num_address_bits=2, shuffle_dataset=True):
        super().__init__(num_address_bits, shuffle_dataset)

    def _create_data(self):
        return list(itertools.product(range(2), repeat=self._total_bits))

    def _create_labels(self, data):
        return [self._multiplexer_func(data_point) for data_point in data]

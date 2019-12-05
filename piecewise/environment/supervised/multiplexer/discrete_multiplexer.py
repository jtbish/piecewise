import itertools

from .abstract_multiplexer import AbstractMultiplexer


class DiscreteMultiplexer(AbstractMultiplexer):
    def __init__(self,
                 num_address_bits=2,
                 shuffle_dataset=True,
                 reward_correct=1000,
                 reward_incorrect=0):
        super().__init__(num_address_bits, shuffle_dataset, reward_correct,
                         reward_incorrect)

    def _create_data(self):
        return list(itertools.product(range(2), repeat=self._total_bits))

    def _create_labels(self, data):
        return [self._multiplexer_func(data_point) for data_point in data]

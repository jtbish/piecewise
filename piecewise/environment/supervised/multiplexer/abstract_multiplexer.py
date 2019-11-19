import abc

import pandas as pd

from ..supervised_environment import SupervisedEnvironment


def calc_total_bits(num_address_bits):
    return num_address_bits + 2**num_address_bits


class AbstractMultiplexer(SupervisedEnvironment, metaclass=abc.ABCMeta):
    def __init__(self, num_address_bits, shuffle_dataset):
        self._num_address_bits = num_address_bits
        self._num_register_bits = 2**num_address_bits
        self._total_bits = calc_total_bits(num_address_bits)

        dataset = self._create_dataset()
        super().__init__(dataset, shuffle_dataset)

    def _multiplexer_func(self, bit_array):
        address_bits = bit_array[0:self._num_address_bits]
        register_bits = bit_array[self._num_address_bits:self._total_bits]
        register_bits_idx = self._get_decimal_value_of_bits(address_bits)
        return register_bits[register_bits_idx]

    def _get_decimal_value_of_bits(self, bits):
        bitstring = "".join([str(bit) for bit in bits])
        decimal_value = int(bitstring, 2)
        return decimal_value

    def _create_dataset(self):
        data = self._create_data()
        labels = self._create_labels(data)
        return self._create_data_frame(data, labels)

    def _create_data_frame(self, data, labels):
        data_frame = pd.DataFrame(data)
        self._rename_feature_columns(data_frame)
        self._append_label_column(data_frame, labels)
        return data_frame

    def _rename_feature_columns(self, data_frame):
        column_names = [
            f"addr_bit{num}" for num in range(1, self._num_address_bits + 1)
        ]
        column_names.extend([
            f"register_bit{num}"
            for num in range(1, self._num_register_bits + 1)
        ])
        data_frame.columns = column_names

    def _append_label_column(self, data_frame, labels):
        data_frame["label"] = labels

    @abc.abstractmethod
    def _create_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _create_labels(self, data):
        raise NotImplementedError

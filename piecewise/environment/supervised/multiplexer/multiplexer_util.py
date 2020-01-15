"""Utility functions for multiplexer builders."""


def calc_num_register_bits(num_address_bits):
    return 2**num_address_bits


def calc_total_bits(num_address_bits):
    return num_address_bits + calc_num_register_bits(num_address_bits)


def multiplexer_func(num_address_bits, bit_array):
    address_bits = bit_array[:num_address_bits]
    register_bits = bit_array[num_address_bits:]
    register_bits_idx = _get_decimal_value_of_bits(address_bits)
    return register_bits[register_bits_idx]


def _get_decimal_value_of_bits(bits):
    bitstring = "".join([str(bit) for bit in bits])
    decimal_value = int(bitstring, 2)
    return decimal_value

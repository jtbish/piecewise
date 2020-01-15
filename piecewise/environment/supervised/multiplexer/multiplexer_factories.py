"""Factory functions for making multiplexer environments."""
from piecewise.error.environment_error import InvalidSpecError

from .multiplexer_builders import (DiscreteMultiplexerBuilder,
                                   MultiplexerDirector, RealMultiplexerBuilder)


def make_discrete_mux_env(num_address_bits=2,
                          shuffle_dataset=True,
                          reward_correct=1000,
                          reward_incorrect=0):
    """Factory function for making a discrete multiplexer environment."""
    num_address_bits = _validate_and_return_num_address_bits(num_address_bits)
    mux_builder = DiscreteMultiplexerBuilder(num_address_bits)
    mux_director = MultiplexerDirector(mux_builder, num_address_bits,
                                       shuffle_dataset, reward_correct,
                                       reward_incorrect)
    return mux_director.make_env()


def make_real_mux_env(thresholds,
                      num_address_bits=2,
                      shuffle_dataset=True,
                      num_samples=1000,
                      seed=0,
                      reward_correct=1000,
                      reward_incorrect=0):
    """Factory function for making a real multiplexer environment."""
    num_address_bits = _validate_and_return_num_address_bits(num_address_bits)
    mux_builder = RealMultiplexerBuilder(num_address_bits, num_samples, seed,
                                         thresholds)
    mux_director = MultiplexerDirector(mux_builder, num_address_bits,
                                       shuffle_dataset, reward_correct,
                                       reward_incorrect)
    return mux_director.make_env()


def _validate_and_return_num_address_bits(num_address_bits):
    num_address_bits = int(num_address_bits)
    if not num_address_bits > 0:
        raise InvalidSpecError("Invalid num address bits for multiplexer:"
                               f"{num_address_bits}, must be positive integer")
    return num_address_bits

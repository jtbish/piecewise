"""Factory functions for making multiplexer environments."""
from piecewise.error.environment_error import InvalidSpecError

from .multiplexer_builders import (DiscreteMultiplexerBuilder,
                                   MultiplexerDirector, RealMultiplexerBuilder)


def make_discrete_mux_env(num_address_bits=2,
                          shuffle_dataset=True,
                          shuffle_seed=0,
                          reward_correct=1000,
                          reward_incorrect=0):
    """Factory function for making a discrete multiplexer environment."""
    num_address_bits = _validate_and_return_num_address_bits(num_address_bits)
    mux_builder = DiscreteMultiplexerBuilder(num_address_bits)
    mux_director = MultiplexerDirector(mux_builder, num_address_bits,
                                       shuffle_dataset, shuffle_seed,
                                       reward_correct, reward_incorrect)
    env = mux_director.make_env()
    env.record_parametrization(name="discrete_mux",
                               num_address_bits=num_address_bits,
                               shuffle_dataset=shuffle_dataset,
                               shuffle_seed=shuffle_seed,
                               reward_correct=reward_correct,
                               reward_incorrect=reward_incorrect)
    return env


def make_real_mux_env(thresholds,
                      num_address_bits=2,
                      shuffle_dataset=True,
                      shuffle_seed=0,
                      num_samples=1000,
                      data_gen_seed=0,
                      reward_correct=1000,
                      reward_incorrect=0):
    """Factory function for making a real multiplexer environment."""
    num_address_bits = _validate_and_return_num_address_bits(num_address_bits)
    mux_builder = RealMultiplexerBuilder(num_address_bits, num_samples,
                                         data_gen_seed, thresholds)
    mux_director = MultiplexerDirector(mux_builder, num_address_bits,
                                       shuffle_dataset, shuffle_seed,
                                       reward_correct, reward_incorrect)
    env = mux_director.make_env()
    # TODO this is gross
    env.record_parametrization(name="real_mux",
                               thresholds=thresholds,
                               num_address_bits=num_address_bits,
                               shuffle_dataset=shuffle_dataset,
                               num_samples=num_samples,
                               data_gen_seed=data_gen_seed,
                               reward_correct=reward_correct,
                               reward_incorrect=reward_incorrect)
    return mux_director.make_env()


def _validate_and_return_num_address_bits(num_address_bits):
    num_address_bits = int(num_address_bits)
    if not num_address_bits > 0:
        raise InvalidSpecError("Invalid num address bits for multiplexer:"
                               f"{num_address_bits}, must be positive integer")
    return num_address_bits

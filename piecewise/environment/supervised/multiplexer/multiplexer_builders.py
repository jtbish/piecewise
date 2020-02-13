"""Director and builder classes for making multiplexer environments."""
import abc
import itertools

import numpy as np
import pandas as pd

from piecewise.dtype import DataSpaceBuilder, Dimension
from piecewise.environment import ClassificationEnvironment
from piecewise.error.environment_error import InvalidSpecError
from piecewise.util.rng import init_np_random_state

from .multiplexer_util import (calc_num_register_bits, calc_total_bits,
                               multiplexer_func)


class MultiplexerDirector:
    """Director for multiplexer builder classes."""
    def __init__(self, mux_builder, num_address_bits, shuffle_dataset,
                 shuffle_seed, reward_correct, reward_incorrect):
        self._mux_builder = mux_builder
        self._num_address_bits = num_address_bits
        self._num_register_bits = \
            calc_num_register_bits(self._num_address_bits)
        self._shuffle_dataset = shuffle_dataset
        self._shuffle_seed = shuffle_seed
        self._reward_correct = reward_correct
        self._reward_incorrect = reward_incorrect

    def make_env(self):
        """Constructs and returns a new ClassificationEnvironment containing
        a multiplexer dataset."""
        data = self._mux_builder.create_data()
        labels = self._mux_builder.create_labels(data)
        dataset = self._create_dataset(data, labels)
        obs_space = self._mux_builder.create_obs_space()
        env = ClassificationEnvironment(
            dataset=dataset,
            obs_space=obs_space,
            action_set=None,
            shuffle_dataset=self._shuffle_dataset,
            shuffle_seed=self._shuffle_seed,
            reward_correct=self._reward_correct,
            reward_incorrect=self._reward_incorrect)
        return env

    def _create_dataset(self, data, labels):
        dataset = pd.DataFrame(data)
        self._rename_feature_columns(dataset)
        self._append_label_column(dataset, labels)
        return dataset

    def _rename_feature_columns(self, dataset):
        column_names = [
            f"addr_bit{num}" for num in range(1, self._num_address_bits + 1)
        ]
        column_names.extend([
            f"register_bit{num}"
            for num in range(1, self._num_register_bits + 1)
        ])
        dataset.columns = column_names

    def _append_label_column(self, dataset, labels):
        dataset["label"] = labels


class IMultiplexerBuilder(metaclass=abc.ABCMeta):
    """Interface for multiplexer builder classes."""
    @abc.abstractmethod
    def create_data(self):
        """Creates the feature data for the multiplexer problem."""
        raise NotImplementedError

    @abc.abstractmethod
    def create_labels(self, data):
        """Creates the labels for the multiplexer problem."""
        raise NotImplementedError

    @abc.abstractmethod
    def create_obs_space(self):
        """Creates the observation space for the multiplexer problem.

        This is only necessary for real multiplexer (as explained in its
        implementation of this method). Discrete multiplexer implementation
        explains why it can do 'nothing'."""
        raise NotImplementedError


class DiscreteMultiplexerBuilder(IMultiplexerBuilder):
    """Concrete builder for discrete multiplexer environments."""
    def __init__(self, num_address_bits):
        self._num_address_bits = num_address_bits
        self._total_bits = calc_total_bits(num_address_bits)

    def create_data(self):
        return list(itertools.product(range(2), repeat=self._total_bits))

    def create_labels(self, data):
        return [
            multiplexer_func(self._num_address_bits, data_point)
            for data_point in data
        ]

    def create_obs_space(self):
        """The ClassificationEnvironment that is constructed
        with the data and labels is able to generate the correct obs space
        automatically, so signal it to do that by giving no obs space."""
        return None


class RealMultiplexerBuilder(IMultiplexerBuilder):
    """Concrete builder for real multiplexer environments."""
    THRESHOLD_MIN = 0.0
    THRESHOLD_MAX = 1.0

    def __init__(self, num_address_bits, num_samples, data_gen_seed,
                 thresholds):
        self._num_address_bits = num_address_bits
        self._num_samples = num_samples
        self._np_random = init_np_random_state(data_gen_seed)
        self._total_bits = calc_total_bits(self._num_address_bits)
        self._thresholds = self._convert_and_validate_thresholds(
            thresholds, self._total_bits)

    def _convert_and_validate_thresholds(self, thresholds, total_bits):
        thresholds = self._convert_thresholds_to_flat_nparray(thresholds)
        self._validate_thresholds(thresholds, total_bits)
        return thresholds

    def _convert_thresholds_to_flat_nparray(self, thresholds):
        return np.asarray(thresholds).flatten()

    def _validate_thresholds(self, thresholds, total_bits):
        are_correct_len = len(thresholds) == total_bits
        are_floats = self._thresholds_are_floats(thresholds)
        are_in_valid_range = \
            self._thresholds_are_in_valid_range(thresholds)
        if not (are_correct_len and are_floats and are_in_valid_range):
            raise InvalidSpecError(
                "Given thresholds for real multiplexer not valid.")

    def _thresholds_are_floats(self, thresholds):
        return np.all(
            [isinstance(elem, np.floating) for elem in thresholds.flatten()])

    def _thresholds_are_in_valid_range(self, thresholds):
        return np.all(self.THRESHOLD_MIN <= thresholds) and \
            np.all(thresholds <= self.THRESHOLD_MAX)

    def create_data(self):
        return self._np_random.rand(self._num_samples, self._total_bits)

    def create_labels(self, data):
        labels = []
        for data_point in data:
            bit_array = self._apply_thresholding_to(data_point)
            labels.append(multiplexer_func(self._num_address_bits, bit_array))
        return labels

    def _apply_thresholding_to(self, data_point):
        bit_array = []
        for feature_val, threshold in zip(data_point, self._thresholds):
            if feature_val < threshold:
                bit_array.append(0)
            else:
                bit_array.append(1)
        return bit_array

    def create_obs_space(self):
        """Need to build obs space for real mux manually because its
        randomly generated data may not cover the entire [0.0, 1.0] range on
        one or all feautre dimensions.

        Automatic obs space generation in ClassificationEnvironment may
        therefore not create the truly correct obs space, hence creating it
        here is necessary."""
        num_features = self._total_bits
        obs_space_builder = DataSpaceBuilder()
        for _ in range(num_features):
            obs_space_builder.add_dim(Dimension(0.0, 1.0))
        return obs_space_builder.create_space()

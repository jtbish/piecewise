import numpy as np

from piecewise.dtype import DataSpaceBuilder, Dimension
from piecewise.error.environment_error import InvalidSpecError

from .abstract_multiplexer import AbstractMultiplexer, calc_total_bits

THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0


class RealMultiplexer(AbstractMultiplexer):
    def __init__(self,
                 num_address_bits=2,
                 shuffle_dataset=True,
                 num_samples=1000,
                 seed=0,
                 thresholds=None,
                 reward_correct=1000,
                 reward_incorrect=0):
        self._num_samples = num_samples
        self._set_seed(seed)
        self._thresholds = self._gen_thresholds(thresholds, num_address_bits)
        super().__init__(num_address_bits, shuffle_dataset, reward_correct,
                         reward_incorrect)

    def _set_seed(self, seed):
        np.random.seed(seed)

    def _gen_obs_space(self, data):
        """Need to build obs space for real mux manually because its
        randomly generated data may not cover the entire [0.0, 1.0] range on
        one or all feautre dimensions.

        Automatic obs space generation in SupervisedEnvironment may therefore
        not create the truly correct obs space, hence overriding in this
        subclass is necessary."""
        num_features = self._total_bits
        obs_space_builder = DataSpaceBuilder()
        for _ in range(num_features):
            obs_space_builder.add_dim(Dimension(0.0, 1.0))
        return obs_space_builder.create_space()

    def _gen_thresholds(self, given_thresholds, num_address_bits):
        total_bits = calc_total_bits(num_address_bits)
        if given_thresholds is None:
            return np.random.random(size=total_bits)
        else:
            given_thresholds = np.asarray(given_thresholds)
            self._validate_thresholds(given_thresholds, total_bits)
            return given_thresholds

    def _validate_thresholds(self, given_thresholds, total_bits):
        are_correct_len = len(given_thresholds) == total_bits
        are_floats = self._thresholds_are_floats(given_thresholds)
        are_in_valid_range = \
            self._thresholds_are_in_valid_range(given_thresholds)
        if not (are_correct_len and are_floats and are_in_valid_range):
            raise InvalidSpecError(
                "Given thresholds for real multiplexer not valid.")

    def _thresholds_are_floats(self, given_thresholds):
        return np.all([
            isinstance(elem, np.floating)
            for elem in given_thresholds.flatten()
        ])

    def _thresholds_are_in_valid_range(self, given_thresholds):
        return np.all(THRESHOLD_MIN <= given_thresholds) and \
            np.all(given_thresholds < THRESHOLD_MAX)

    def _create_data(self):
        return np.random.random(size=(self._num_samples, self._total_bits))

    def _create_labels(self, data):
        labels = []
        for data_point in data:
            bit_array = self._apply_thresholding_to(data_point)
            labels.append(self._multiplexer_func(bit_array))
        return labels

    def _apply_thresholding_to(self, data_point):
        bit_array = []
        for feature_val, threshold in zip(data_point, self._thresholds):
            if feature_val < threshold:
                bit_array.append(0)
            else:
                bit_array.append(1)
        return bit_array

    @property
    def thresholds(self):
        return self._thresholds

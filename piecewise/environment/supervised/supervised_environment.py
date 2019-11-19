import numpy as np
import pandas as pd

from piecewise.dtype import DataSpaceBuilder, Dimension

from ..environment import Environment, EnvironmentResponse, check_terminal

REWARD_CORRECT = 1000
REWARD_INCORRECT = 0


class SupervisedEnvironment(Environment):
    def __init__(self, dataset, shuffle_dataset=True):
        self._dataset = self._try_convert_to_data_frame(dataset)
        self._shuffle_dataset = shuffle_dataset
        self._num_data_points = self._dataset.shape[0]
        self._num_features = self._dataset.shape[1] - 1
        data, labels = self._split_dataset()
        obs_space = self._gen_obs_space(data)
        action_set = self._gen_action_set(labels)
        super().__init__(obs_space, action_set)

    def _try_convert_to_data_frame(self, dataset):
        is_already_data_frame = isinstance(dataset, pd.DataFrame)
        if is_already_data_frame:
            # assume caller set up ok
            return dataset
        else:
            return self._create_data_frame(dataset)

    def _create_data_frame(self, dataset):
        data_frame = pd.DataFrame(dataset)
        self._rename_data_frame_columns(data_frame)
        return data_frame

    def _rename_data_frame_columns(self, data_frame):
        generic_column_names = [
            f"feature{feature_num}"
            for feature_num in range(1, self._num_features + 1)
        ]
        generic_column_names.append("label")
        data_frame.columns = generic_column_names

    def _gen_obs_space(self, data):
        obs_space_builder = DataSpaceBuilder()
        for column_idx in range(self._num_features):
            feature_vec = data.iloc[:, column_idx]
            obs_space_builder.add_dim(self._make_obs_space_dim(feature_vec))
        return obs_space_builder.create_space()

    def _make_obs_space_dim(self, feature_vec):
        lower = np.min(feature_vec)
        upper = np.max(feature_vec)
        return Dimension(lower, upper)

    def _gen_action_set(self, labels):
        return set([label for label in labels])

    def reset(self):
        self._dataset_idx_order = self._choose_dataset_idx_order()
        self._idx_into_dataset_idx_order = 0

    def _choose_dataset_idx_order(self):
        if self._shuffle_dataset:
            return np.random.permutation(range(self._num_data_points))
        else:
            return list(range(self._num_data_points))

    def _split_dataset(self):
        data = self._dataset.iloc[:, 0:-1]
        labels = self._dataset.iloc[:, -1]
        return data, labels

    @check_terminal
    def observe(self):
        data, _ = self._split_dataset()
        data_idx = self._get_dataset_idx()
        data_point = data.iloc[data_idx, :]
        return np.asarray(data_point)

    @check_terminal
    def act(self, action):
        given_label = action

        _, labels = self._split_dataset()
        label_idx = self._get_dataset_idx()
        actual_label = labels.iloc[label_idx]
        self._idx_into_dataset_idx_order += 1

        is_correct_label = given_label == actual_label
        reward = self._calc_reward(is_correct_label)
        return EnvironmentResponse(reward, is_correct_label)

    def _get_dataset_idx(self):
        return self._dataset_idx_order[self._idx_into_dataset_idx_order]

    def _calc_reward(self, is_correct_label):
        if is_correct_label:
            return REWARD_CORRECT
        else:
            return REWARD_INCORRECT

    def is_terminal(self):
        return self._idx_into_dataset_idx_order == self._num_data_points

    @property
    def dataset(self):
        return self._dataset

    @property
    def data(self):
        data, _ = self._split_dataset()
        return np.asarray(data)

    @property
    def labels(self):
        _, labels = self._split_dataset()
        return np.asarray(labels)

import numpy as np
import pandas as pd

from piecewise.dtype import DataSpaceBuilder, Dimension
from piecewise.util.rng import init_np_random_state

from ..environment import (EnvironmentABC, EnvironmentResponse,
                           EnvironmentStepTypes, check_terminal)


class ClassificationEnvironment(EnvironmentABC):
    """Environment that manages interaction with a static labelled dataset."""
    def __init__(self,
                 dataset,
                 obs_space=None,
                 action_set=None,
                 shuffle_dataset=True,
                 shuffle_seed=0,
                 reward_correct=1000,
                 reward_incorrect=0):
        self._dataset = self._format_dataset_if_necessary(dataset)
        self._shuffle_dataset = bool(shuffle_dataset)
        self._np_random = init_np_random_state(shuffle_seed)
        self._reward_correct = float(reward_correct)
        self._reward_incorrect = float(reward_incorrect)
        self._num_data_points = self._dataset.shape[0]
        self._num_features = self._dataset.shape[1] - 1
        data, labels = self._split_dataset()
        obs_space = self._gen_obs_space_if_not_given(data, obs_space)
        action_set = self._gen_action_set_if_not_given(labels, action_set)
        step_type = EnvironmentStepTypes.single_step
        super().__init__(obs_space, action_set, step_type)

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

    def _format_dataset_if_necessary(self, dataset):
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

    def _gen_obs_space_if_not_given(self, data, obs_space):
        if obs_space is None:
            return self._gen_obs_space(data)
        else:
            return obs_space

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

    def _gen_action_set_if_not_given(self, labels, action_set):
        if action_set is None:
            return set([label for label in labels])
        else:
            return action_set

    def reset(self):
        self._dataset_idx_order = self._choose_dataset_idx_order()
        self._idx_into_dataset_idx_order = 0

    def _choose_dataset_idx_order(self):
        if self._shuffle_dataset:
            return self._np_random.permutation(range(self._num_data_points))
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
        return EnvironmentResponse(reward=reward,
                                   was_correct_action=is_correct_label,
                                   is_terminal=self.is_terminal())

    def _get_dataset_idx(self):
        return self._dataset_idx_order[self._idx_into_dataset_idx_order]

    def _calc_reward(self, is_correct_label):
        if is_correct_label:
            return self._reward_correct
        else:
            return self._reward_incorrect

    def is_terminal(self):
        return self._idx_into_dataset_idx_order == self._num_data_points

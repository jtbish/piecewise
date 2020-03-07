import math

import numpy as np

from piecewise.dtype import DataSpaceBuilder, Dimension

from ..environment import EnvironmentResponse, IEnvironment
from .gym_environment import GymEnvironment


def make_frozen_lake_4x4_env(seed=0):
    gym_env = GymEnvironment(env_name="FrozenLake-v0",
                             custom_obs_space=None,
                             custom_action_set=None,
                             seed=seed)
    return FrozenLakeGymEnvironment(gym_env, grid_size=4)


def make_frozen_lake_8x8_env(seed=0):
    gym_env = GymEnvironment(env_name="FrozenLake8x8-v0",
                             custom_obs_space=None,
                             custom_action_set=None,
                             seed=seed)
    return FrozenLakeGymEnvironment(gym_env, grid_size=8)


def make_frozen_lake_custom_env():
    pass


def make_frozen_lake_random_env():
    pass


class FrozenLakeGymEnvironment(IEnvironment):
    """Decorator over GymEnvironment containing frozen lake to change the
    observations and observation space to be an (x, y) grid instead of simple
    numbered array of cells."""
    def __init__(self, gym_env, grid_size):
        assert isinstance(gym_env, GymEnvironment)
        self._raw_env = gym_env
        self._grid_size = grid_size
        self._x_y_coordinates_obs_space = \
            self._gen_x_y_coordinates_obs_space(self._grid_size)

    def _gen_x_y_coordinates_obs_space(self, grid_size):
        obs_space_builder = DataSpaceBuilder()
        for _ in range(2):  # x, y
            obs_space_builder.add_dim(Dimension(0, grid_size - 1))
        return obs_space_builder.create_space()

    @property
    def obs_space(self):
        return self._x_y_coordinates_obs_space

    @property
    def action_set(self):
        return self._raw_env.action_set

    @property
    def step_type(self):
        return self._raw_env.step_type

    def reset(self):
        raw_obs = self._raw_env.reset()
        return self._convert_raw_obs_to_x_y_coordinates(raw_obs)

    def _convert_raw_obs_to_x_y_coordinates(self, raw_obs):
        # raw obs is number indicating idx into flattened grid, where 0 is top
        # left, and flattening is done left to right, top to bottom.
        # x is the column coordinate, y is the row coordinate, both starting
        # from 0.
        assert len(raw_obs) == 1
        obs_val = raw_obs[0]
        x = obs_val % self._grid_size
        y = math.floor(obs_val / self._grid_size)
        assert (y * self._grid_size + x) == obs_val
        return np.asarray([x, y])

    def step(self, action):
        raw_response = self._raw_env.step(action)
        return EnvironmentResponse(
            obs=self._convert_raw_obs_to_x_y_coordinates(raw_response.obs),
            reward=raw_response.reward,
            was_correct_action=raw_response.was_correct_action,
            is_terminal=raw_response.is_terminal)

    def is_terminal(self):
        return self._raw_env.is_terminal()

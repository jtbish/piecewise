import copy
import math

import numpy as np

from gym.envs.registration import register
from gym.envs.toy_text.frozen_lake import MAPS
from piecewise.dtype import DataSpaceBuilder, Dimension

from ..environment import EnvironmentResponse, IEnvironment
from .gym_environment import GymEnvironment

# result of calling generate_random_map in gym frozen_lake.py
MAPS["5x5"] = \
    ["SFFFF",
     "HFHFF",
     "FFFHF",
     "HHFFF",
     "FFHFG"]

MAPS["6x6"] = \
    ["SFFFHH",
     "HHFFHF",
     "HFFFFF",
     "FHFFFH",
     "FFFFHF",
     "FFFFFG"]

register(id="FrozenLake5x5-v0",
         entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"map_name": "5x5"})

register(id="FrozenLake6x6-v0",
         entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"map_name": "6x6"})


def make_frozen_lake_4x4_env(slip_prob=0.0, seed=0):
    is_slippery = slip_prob > 0.0
    gym_env = GymEnvironment(env_name="FrozenLake-v0",
                             env_kwargs={"is_slippery": is_slippery},
                             custom_obs_space=None,
                             custom_action_set=None,
                             seed=seed)
    return FrozenLakeGymEnvironment(gym_env, grid_size=4, slip_prob=slip_prob)


def make_frozen_lake_5x5_env(slip_prob=0.0, seed=0):
    is_slippery = slip_prob > 0.0
    gym_env = GymEnvironment(env_name="FrozenLake5x5-v0",
                             env_kwargs={"is_slippery": is_slippery},
                             custom_obs_space=None,
                             custom_action_set=None,
                             seed=seed)
    return FrozenLakeGymEnvironment(gym_env, grid_size=5, slip_prob=slip_prob)


def make_frozen_lake_6x6_env(slip_prob=0.0, seed=0):
    is_slippery = slip_prob > 0.0
    gym_env = GymEnvironment(env_name="FrozenLake6x6-v0",
                             env_kwargs={"is_slippery": is_slippery},
                             custom_obs_space=None,
                             custom_action_set=None,
                             seed=seed)
    return FrozenLakeGymEnvironment(gym_env, grid_size=6, slip_prob=slip_prob)


def make_frozen_lake_8x8_env(slip_prob=0.0, seed=0):
    is_slippery = slip_prob > 0.0
    gym_env = GymEnvironment(env_name="FrozenLake8x8-v0",
                             env_kwargs={"is_slippery": is_slippery},
                             custom_obs_space=None,
                             custom_action_set=None,
                             seed=seed)
    return FrozenLakeGymEnvironment(gym_env, grid_size=8, slip_prob=slip_prob)


class FrozenLakeGymEnvironment(IEnvironment):
    """Decorator over GymEnvironment containing frozen lake to change the
    observations and observation space to be an (x, y) grid instead of simple
    numbered array of cells."""
    def __init__(self, gym_env, grid_size, slip_prob):
        assert isinstance(gym_env, GymEnvironment)
        self._raw_env = gym_env
        self._grid_size = grid_size
        self._x_y_coordinates_obs_space = \
            self._gen_x_y_coordinates_obs_space(self._grid_size)
        self._slip_prob = slip_prob
        self._alter_transition_func_if_needed(self._slip_prob)

    def _gen_x_y_coordinates_obs_space(self, grid_size):
        obs_space_builder = DataSpaceBuilder()
        for _ in range(2):  # x, y
            obs_space_builder.add_dim(Dimension(0, grid_size - 1))
        return obs_space_builder.create_space()

    def _alter_transition_func_if_needed(self, slip_prob):
        if slip_prob > 0.0:
            self._alter_transition_func(slip_prob)

    def _alter_transition_func(self, slip_prob):
        assert 0.0 < slip_prob <= 1.0
        # slip prob is 2/3 by default - very high!
        P = self._raw_env._wrapped_env.P
        P_mut = copy.deepcopy(P)
        for state in range(self._raw_env._wrapped_env.nS):
            for action in range(self._raw_env._wrapped_env.nA):
                P_cell_raw = P[state][action]
                is_slippery_transition = len(P_cell_raw) == 3
                if is_slippery_transition:
                    # middle tuple is the desired location, first
                    # and last are non-desired locations
                    (_, ns_1, r_1, done_1) = P_cell_raw[0]
                    (_, ns_2, r_2, done_2) = P_cell_raw[1]
                    (_, ns_3, r_3, done_3) = P_cell_raw[2]
                    prob_non_desired = slip_prob / 2
                    prob_desired = (1 - slip_prob)
                    P_cell_mut = []
                    if prob_non_desired != 0.0:
                        P_cell_mut.append(
                            (prob_non_desired, ns_1, r_1, done_1))
                        P_cell_mut.append((prob_desired, ns_2, r_2, done_2))
                        P_cell_mut.append(
                            (prob_non_desired, ns_3, r_3, done_3))
                    else:
                        P_cell_mut.append((prob_desired, ns_2, r_2, done_2))
                else:
                    P_cell_mut = P_cell_raw
                P_mut[state][action] = P_cell_mut
        self._raw_env._wrapped_env.unwrapped.P = P_mut

    @property
    def obs_space(self):
        return self._x_y_coordinates_obs_space

    @property
    def action_set(self):
        return self._raw_env.action_set

    @property
    def step_type(self):
        return self._raw_env.step_type

    @property
    def P(self):
        return self._raw_env._wrapped_env.P

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def slip_prob(self):
        return self._slip_prob

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

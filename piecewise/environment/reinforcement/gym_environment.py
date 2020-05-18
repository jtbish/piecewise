import numpy as np

import gym
from gym.spaces import Box, Discrete
from piecewise.dtype import DataSpaceBuilder, Dimension
from piecewise.error.core_errors import InternalError

from ..environment import (CorrectActionNotApplicable, EnvironmentResponse,
                           EnvironmentStepTypes, IEnvironment, check_terminal)


class GymEnvironment(IEnvironment):
    """Wrapper over an OpenAI Gym environment to conform with Piecewise
    environment interface.

    Supports discrete / continuous obs space, discrete action set."""
    def __init__(self,
                 env_name,
                 env_kwargs=None,
                 custom_obs_space=None,
                 custom_action_set=None,
                 seed=0):
        self._wrapped_env = self._init_wrapped_env(env_name, env_kwargs, seed)
        self._obs_space = self._gen_obs_space_if_not_given(
            self._wrapped_env, custom_obs_space)
        self._action_set = self._gen_action_set_if_not_given(
            self._wrapped_env, custom_action_set)
        self._is_terminal = True

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_set(self):
        return self._action_set

    @property
    def step_type(self):
        return EnvironmentStepTypes.multi_step

    def _init_wrapped_env(self, env_name, env_kwargs, seed):
        if env_kwargs is None:
            env_kwargs = {}
        wrapped_env = gym.make(env_name, **env_kwargs)
        wrapped_env.seed(seed)
        return wrapped_env

    def _gen_obs_space_if_not_given(self, wrapped_env, custom_obs_space):
        if custom_obs_space is None:
            return self._gen_obs_space(wrapped_env)
        else:
            return custom_obs_space

    def _gen_obs_space(self, wrapped_env):
        if isinstance(wrapped_env.observation_space, Discrete):
            return self._gen_discrete_obs_space(wrapped_env)
        elif isinstance(wrapped_env.observation_space, Box):
            return self._gen_continuous_obs_space(wrapped_env)
        else:
            raise InternalError("Unrecognised gym environment obs space type.")

    def _gen_discrete_obs_space(self, wrapped_env):
        num_obss = wrapped_env.observation_space.n
        obs_space_builder = DataSpaceBuilder()
        obs_space_builder.add_dim(Dimension(lower=0, upper=(num_obss - 1)))
        return obs_space_builder.create_space()

    def _gen_continuous_obs_space(self, wrapped_env):
        lower_vector = wrapped_env.observation_space.low
        upper_vector = wrapped_env.observation_space.high
        obs_space_builder = DataSpaceBuilder()
        for (lower, upper) in zip(lower_vector, upper_vector):
            obs_space_builder.add_dim(Dimension(lower, upper))
        return obs_space_builder.create_space()

    def _gen_action_set_if_not_given(self, wrapped_env, custom_action_set):
        if custom_action_set is None:
            return self._gen_action_set(wrapped_env)
        else:
            return custom_action_set

    def _gen_action_set(self, wrapped_env):
        num_actions = wrapped_env.action_space.n
        return set(range(num_actions))

    def reset(self):
        self._is_terminal = False
        wrapped_obs = self._wrapped_env.reset()
        return self._enforce_valid_obs(wrapped_obs)

    def _enforce_valid_obs(self, obs):
        obs = np.atleast_1d(obs)
        obs = self._truncate_obs(obs)
        return obs

    def _truncate_obs(self, obs):
        """Necessary to enforce observations are in (possibly) custom obs
        space."""
        truncated_obs = []
        for (feature_val, dimension) in zip(obs, self._obs_space):
            feature_val = max(feature_val, dimension.lower)
            feature_val = min(feature_val, dimension.upper)
            truncated_obs.append(feature_val)
        return np.asarray(truncated_obs)

    @check_terminal
    def step(self, action):
        assert action in self._action_set
        wrapped_obs, wrapped_reward, wrapped_done, wrapped_info = \
            self._wrapped_env.step(action)
        self._is_terminal = wrapped_done
        return EnvironmentResponse(
            obs=self._enforce_valid_obs(wrapped_obs),
            reward=wrapped_reward,
            was_correct_action=CorrectActionNotApplicable,
            is_terminal=self._is_terminal)

    def is_terminal(self):
        return self._is_terminal

    def render(self):
        self._wrapped_env.render()


class NormalisedGymEnvironment(IEnvironment):
    """Decorator obj for GymEnvironment operating in continuous space to
    normalise the observation space."""
    def __init__(self, gym_env):
        assert isinstance(gym_env, GymEnvironment)
        self._raw_env = gym_env
        self._unit_hypercube_obs_space = \
            self._gen_unit_hypercube_obs_space(
                    num_dimensions=len(self._raw_env.obs_space))

    def _gen_unit_hypercube_obs_space(self, num_dimensions):
        obs_space_builder = DataSpaceBuilder()
        for _ in range(num_dimensions):
            obs_space_builder.add_dim(Dimension(0.0, 1.0))
        return obs_space_builder.create_space()

    @property
    def obs_space(self):
        return self._unit_hypercube_obs_space

    @property
    def action_set(self):
        return self._raw_env.action_set

    @property
    def step_type(self):
        return self._raw_env.step_type

    def reset(self):
        raw_obs = self._raw_env.reset()
        return self._normalise_raw_obs(raw_obs)

    def _normalise_raw_obs(self, raw_obs):
        normalised_obs = []
        for (raw_obs_val, raw_obs_space_dim) in \
                zip(raw_obs, self._raw_env.obs_space):
            normalised_val = (raw_obs_val - raw_obs_space_dim.lower) / \
                (raw_obs_space_dim.upper - raw_obs_space_dim.lower)
            assert 0.0 <= normalised_val <= 1.0
            normalised_obs.append(normalised_val)

        return np.asarray(normalised_obs)

    def step(self, action):
        raw_response = self._raw_env.step(action)
        return EnvironmentResponse(
            obs=self._normalise_raw_obs(raw_response.obs),
            reward=raw_response.reward,
            was_correct_action=raw_response.was_correct_action,
            is_terminal=raw_response.is_terminal)

    def is_terminal(self):
        return self._raw_env.is_terminal()

    def render(self):
        return self._raw_env.render()

import gym
import numpy as np
from gym import logger

from piecewise.dtype import DataSpaceBuilder, Dimension

from ..environment import (CorrectActionNotApplicable, EnvironmentResponse,
                           EnvironmentStepTypes, IEnvironment, check_terminal)

# quieten warnings from gym
gym.logger.set_level(logger.ERROR)


class GymEnvironment(IEnvironment):
    """Wrapper over an OpenAI Gym environment to conform with Piecewise
    environment interface."""
    def __init__(self,
                 env_name,
                 custom_obs_space=None,
                 custom_action_set=None,
                 seed=0):
        self._wrapped_env = self._init_wrapped_env(env_name, seed)
        self._obs_space = self._gen_obs_space_if_not_given(
            self._wrapped_env, custom_obs_space)
        self._action_set = self._gen_action_set_if_not_given(
            self._wrapped_env, custom_action_set)
        self._curr_obs = None
        self.reset()

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_set(self):
        return self._action_set

    @property
    def step_type(self):
        return EnvironmentStepTypes.multi_step

    def _init_wrapped_env(self, env_name, seed):
        wrapped_env = gym.make(env_name)
        wrapped_env.seed(seed)
        return wrapped_env

    def _gen_obs_space_if_not_given(self, wrapped_env, custom_obs_space):
        if custom_obs_space is None:
            return self._gen_obs_space(wrapped_env)
        else:
            return custom_obs_space

    def _gen_obs_space(self, wrapped_env):
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
        self._curr_obs = self._wrapped_env.reset()
        self._is_terminal = False
        self._wrapped_env_was_done_last_step = False

    @check_terminal
    def observe(self):
        obs = self._curr_obs
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
        return truncated_obs

    @check_terminal
    def act(self, action):
        assert action in self._action_set
        wrapped_obs, wrapped_reward, wrapped_done, _ = self._wrapped_env.step(
            action)
        self._curr_obs = wrapped_obs
        self._is_terminal = self._wrapped_env_was_done_last_step
        self._wrapped_env_was_done_last_step = wrapped_done
        return EnvironmentResponse(
            reward=wrapped_reward,
            was_correct_action=CorrectActionNotApplicable,
            is_terminal=self._is_terminal)

    def is_terminal(self):
        return self._is_terminal


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
        self._raw_env.reset()

    def observe(self):
        raw_obs = self._raw_env.observe()
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

    def act(self, action):
        return self._raw_env.act(action)

    def is_terminal(self):
        return self._raw_env.is_terminal()

class DiscretisedGymEnvironment(IEnvironment):
    """Decorator obj for GymEnvironment operating in continuous space to
    discretise the observation space."""
    def __init__(self, gym_env, discertisation_vec):
        assert isinstance(gym_env, GymEnvironment)
        self._raw_env = gym_env
        self._discretisation_vec = discertisation_vec
        self._discretised_obs_space = self._gen_discretised_obs_space(
                discertisation_vec)

    def _gen_discretised_obs_space(self, discertisation_vec):
        obs_space_builder = DataSpaceBuilder()
        for num_buckets in discertisation_vec:
            obs_space_builder.add_dim(Dimension(0.0, 1.0))
        return obs_space_builder.create_space()

    @property
    def obs_space(self):
        return self._discretised_obs_space

    @property
    def action_set(self):
        return self._raw_env.action_set

    @property
    def step_type(self):
        return self._raw_env.step_type

    def reset(self):
        self._raw_env.reset()

    def observe(self):
        raw_obs = self._raw_env.observe()
        return self._discretise_raw_obs(raw_obs)

    def _discretise_raw_obs(self, raw_obs):
        discretised_obs = []
        for (raw_obs_val, raw_obs_space_dim) in \
                zip(raw_obs, self._raw_env.obs_space):
            discretised_val = (raw_obs_val - raw_obs_space_dim.lower) / \
                (raw_obs_space_dim.upper - raw_obs_space_dim.lower)
            assert 0.0 <= discretised_val <= 1.0
            discretised_obs.append(discretised_val)

        return np.asarray(discretised_obs)

    def act(self, action):
        return self._raw_env.act(action)

    def is_terminal(self):
        return self._raw_env.is_terminal()

import gym
from gym import logger

from piecewise.dtype import DataSpaceBuilder, Dimension

from ..environment import (CorrectActionNotApplicable, EnvironmentABC,
                           EnvironmentResponse, EnvironmentStepTypes,
                           check_terminal)

# quieten warnings from gym
gym.logger.set_level(logger.ERROR)


class GymEnvironment(EnvironmentABC):
    """Wrapper over an OpenAI Gym environment to conform with Piecewise
    environment interface."""
    def __init__(self,
                 env_name,
                 custom_obs_space=None,
                 custom_action_set=None,
                 seed=0,
                 normalise=False):
        self._wrapped_env = self._init_wrapped_env(env_name, seed)
        self._do_normalisation = normalise
        self._internal_obs_space = self._gen_internal_obs_space_if_not_given(
            self._wrapped_env, custom_obs_space)
        self._client_obs_space = self._create_client_obs_space(
            self._internal_obs_space, self._do_normalisation)
        action_set = self._gen_action_set_if_not_given(self._wrapped_env,
                                                       custom_action_set)
        self._curr_obs = None
        step_type = EnvironmentStepTypes.multi_step
        super().__init__(action_set=action_set, step_type=step_type)

    @property
    def obs_space(self):
        return self._client_obs_space

    def _init_wrapped_env(self, env_name, seed):
        wrapped_env = gym.make(env_name)
        wrapped_env.seed(seed)
        return wrapped_env

    def _gen_internal_obs_space_if_not_given(self, wrapped_env,
                                             custom_obs_space):
        if custom_obs_space is None:
            return self._gen_internal_obs_space(wrapped_env)
        else:
            return custom_obs_space

    def _gen_internal_obs_space(self, wrapped_env):
        lower_vector = wrapped_env.observation_space.low
        upper_vector = wrapped_env.observation_space.high
        obs_space_builder = DataSpaceBuilder()
        for (lower, upper) in zip(lower_vector, upper_vector):
            obs_space_builder.add_dim(Dimension(lower, upper))
        return obs_space_builder.create_space()

    def _create_client_obs_space(self, internal_obs_space, do_normalisation):
        if do_normalisation:
            unit_hypercube = DataSpaceBuilder()
            for _ in range(len(internal_obs_space)):
                unit_hypercube.add_dim(Dimension(0.0, 1.0))
            return unit_hypercube.create_space()
        else:
            return internal_obs_space

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
        obs = self._normalise_obs_if_needed(obs)
        return obs

    def _truncate_obs(self, obs):
        truncated_obs = []
        for (feature_val, dimension) in zip(obs, self._internal_obs_space):
            feature_val = max(feature_val, dimension.lower)
            feature_val = min(feature_val, dimension.upper)
            truncated_obs.append(feature_val)
        return truncated_obs

    def _normalise_obs_if_needed(self, obs):
        if self._do_normalisation:
            normalised_obs = []
            for (feature_val, dimension) in zip(obs, self._internal_obs_space):
                feature_val = (feature_val - dimension.lower) / (
                    dimension.upper - dimension.lower)
                normalised_obs.append(feature_val)
            return normalised_obs
        else:
            return obs

    @check_terminal
    def act(self, action):
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

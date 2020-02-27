import functools
import math

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
    def __init__(self, env_name, obs_space=None, action_set=None, seed=0):
        self._wrapped_env = self._init_wrapped_env(env_name, seed)
        obs_space = self._gen_obs_space_if_not_given(obs_space)
        action_set = self._gen_action_set_if_not_given(action_set)
        self._curr_obs = None
        step_type = EnvironmentStepTypes.multi_step
        super().__init__(obs_space, action_set, step_type)

    def _init_wrapped_env(self, env_name, seed):
        wrapped_env = gym.make(env_name)
        wrapped_env.seed(seed)
        return wrapped_env

    def _gen_obs_space_if_not_given(self, obs_space):
        if obs_space is None:
            return self._gen_obs_space()
        else:
            return obs_space

    def _gen_obs_space(self):
        lower_vector = self._wrapped_env.observation_space.low
        upper_vector = self._wrapped_env.observation_space.high
        obs_space_builder = DataSpaceBuilder()
        for (lower, upper) in zip(lower_vector, upper_vector):
            obs_space_builder.add_dim(Dimension(lower, upper))
        return obs_space_builder.create_space()

    def _gen_action_set_if_not_given(self, action_set):
        if action_set is None:
            return self._gen_action_set()
        else:
            return action_set

    def _gen_action_set(self):
        num_actions = self._wrapped_env.action_space.n
        return set(range(num_actions))

    def reset(self):
        self._curr_obs = self._wrapped_env.reset()
        self._is_terminal = False
        self._wrapped_env_was_done_last_step = False

    def observe(self):
        return self._curr_obs

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


def discretise_gym_environment(gym_env, discretisation_vec):
    """Discretise GymEnvironment obj at run time."""
    # add new attrs
    gym_env._discretisation_vec = discretisation_vec
    gym_env._orig_obs_space = gym_env._obs_space

    def make_discrete_obs_space(discretisation_vec):
        discrete_obs_space_builder = DataSpaceBuilder()
        for num_bins in discretisation_vec:
            discrete_obs_space_builder.add_dim(
                Dimension(lower=0, upper=(num_bins - 1)))
        return discrete_obs_space_builder.create_space()

    # override obs_space attr so it is discrete
    gym_env._obs_space = make_discrete_obs_space(discretisation_vec)

    # decorate observe() func to do discretisation
    def discretise_observe(observe_method):
        @functools.wraps(observe_method)
        def _discretise_observe(self):
            undiscretised_obs = observe_method()
            assert len(self._discretisation_vec) == len(undiscretised_obs)
            discretised_obs = []
            for (feature_val, orig_obs_space_dim,
                 num_bins) in zip(undiscretised_obs, self._orig_obs_space,
                                  self._discretisation_vec):
                feature_dist_from_dim_lower = \
                    feature_val - orig_obs_space_dim.lower
                assert feature_dist_from_dim_lower >= 0
                bin_width = \
                    (orig_obs_space_dim.upper - orig_obs_space_dim.lower) / \
                    num_bins
                discrete_feature_val = \
                    math.floor(feature_dist_from_dim_lower/bin_width)
                discretised_obs.append(discrete_feature_val)
            return discretised_obs
        return _discretise_observe

    # TODO wtf
    func = discretise_observe(gym_env.observe)
    bound_method = func.__get__(gym_env, gym_env.__class__)
    setattr(gym_env, "observe", bound_method)
    return gym_env

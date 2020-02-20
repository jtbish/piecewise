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
            # TODO extract function
            lower_vector = self._wrapped_env.observation_space.low
            upper_vector = self._wrapped_env.observation_space.high
            obs_space_builder = DataSpaceBuilder()
            for (lower, upper) in zip(lower_vector, upper_vector):
                obs_space_builder.add_dim(Dimension(lower, upper))
            return obs_space_builder.create_space()
        else:
            return obs_space

    def _gen_action_set_if_not_given(self, action_set):
        if action_set is None:
            # TODO extract function
            num_actions = self._wrapped_env.action_space.n
            return set(range(num_actions))
        else:
            return action_set

    def reset(self):
        self._curr_obs = self._wrapped_env.reset()
        self._is_terminal = False
        self._wrapped_env_was_done_last_step = False

    @check_terminal
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

import gym
from gym import logger

from piecewise.dtype import DataSpaceBuilder, Dimension

from ..environment import (CorrectActionNotApplicable, EnvironmentResponse,
                           check_terminal)
from .abstract_reinforcement_environment import \
    AbstractReinforcementEnvironment

# quieten warnings from gym
gym.logger.set_level(logger.ERROR)


class GymEnvironment(AbstractReinforcementEnvironment):
    """Wrapper over an OpenAI Gym environment to conform with Piecewise
    Environment API."""
    def __init__(self, env_name, seed):
        self._wrapped_env = gym.make(env_name)
        obs_space = self._gen_obs_space()
        action_set = self._gen_action_set()
        self._seed = seed
        self._curr_obs = None
        super().__init__(obs_space, action_set)

    def _gen_obs_space(self):
        lower_vector = self._wrapped_env.observation_space.low
        upper_vector = self._wrapped_env.observation_space.high
        obs_space_builder = DataSpaceBuilder()
        for (lower, upper) in zip(lower_vector, upper_vector):
            obs_space_builder.add_dim(Dimension(lower, upper))
        return obs_space_builder.create_space()

    def _gen_action_set(self):
        num_actions = self._wrapped_env.action_space.n
        return set(range(num_actions))

    def reset(self):
        self._wrapped_env.seed(self._seed)
        self._curr_obs = self._wrapped_env.reset()
        self._is_terminal = False
        self._wrapped_env_was_done_last_step = False

    @check_terminal
    def observe(self):
        return self._curr_obs

    @check_terminal
    def act(self, action):
        obs, reward, done, _ = self._wrapped_env.step(action)
        self._curr_obs = obs
        self._is_terminal = self._wrapped_env_was_done_last_step
        self._wrapped_env_was_done_last_step = done
        return EnvironmentResponse(
            reward=reward,
            was_correct_action=CorrectActionNotApplicable,
            is_terminal=self._is_terminal)

    def is_terminal(self):
        return self._is_terminal

import gym
import numpy as np
import pytest

from piecewise.environment import GymEnvironment

_GYM_ENV_NAME = "CartPole-v0"
_ENV_SEED = 0
_DUMMY_ACTION = 0


@pytest.fixture
def wrapper_env():
    return GymEnvironment(_GYM_ENV_NAME, _ENV_SEED)


@pytest.fixture
def wrapped_env():
    env = gym.make(_GYM_ENV_NAME)
    env.seed(_ENV_SEED)
    return env


class TestGymEnvironment:
    def test_obs_seq_is_same(self, wrapped_env, wrapper_env):
        first_wrapped_env_obs = wrapped_env.reset()
        expected_obs_seq = [first_wrapped_env_obs]
        while True:
            obs, _, done, _ = wrapped_env.step(_DUMMY_ACTION)
            expected_obs_seq.append(obs)
            if done:
                break

        actual_obs_seq = []
        while not wrapper_env.is_terminal():
            obs = wrapper_env.observe()
            actual_obs_seq.append(obs)
            wrapper_env.act(_DUMMY_ACTION)

        assert np.array_equal(expected_obs_seq, actual_obs_seq)

    def test_return_is_same(self, wrapped_env, wrapper_env):
        wrapped_env.reset()
        wrapped_return = 0
        while True:
            _, reward, done, _ = wrapped_env.step(_DUMMY_ACTION)
            wrapped_return += reward
            if done:
                break

        wrapper_return = 0
        while not wrapper_env.is_terminal():
            env_response = wrapper_env.act(_DUMMY_ACTION)
            wrapper_return += env_response.reward

        assert wrapped_return == wrapper_return

    def test_is_terminal_after_same_trajectory(self, wrapped_env, wrapper_env):
        # TODO
        pass

    def test_same_obs_on_repeated_observe(self):
        # TODO
        pass

    def test_out_of_data_on_extra_act(self):
        # TODO
        pass

    def test_out_of_data_on_extra_observe(self):
        # TODO
        pass

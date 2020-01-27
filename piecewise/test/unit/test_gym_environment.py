import gym
import numpy as np
import pytest

from piecewise.environment import EnvironmentStepTypes, GymEnvironment
from piecewise.error.environment_error import OutOfDataError

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
    """Since GymEnvironment is a wrapper over a native (wrapped) gym env, the
    behaviour of the wrapper env should be consistent with the behaviour of the
    wrapped env.

    Thus, the strategy for testing is to perform some process on both envs
    (using their slighlty different APIs), and asserting the output of the
    process is the same."""
    def test_step_type(self, wrapper_env):
        assert wrapper_env.step_type == EnvironmentStepTypes.multi_step

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

    def test_undiscounted_return_is_same(self, wrapped_env, wrapper_env):
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

    def test_same_obs_on_repeated_observe(self, wrapper_env):
        last_obs = None

        for _ in range(2):
            obs = wrapper_env.observe()
            if last_obs is not None:
                assert np.array_equal(last_obs, obs)
            last_obs = obs

    def test_out_of_data_on_extra_act(self, wrapper_env):
        while not wrapper_env.is_terminal():
            wrapper_env.act(_DUMMY_ACTION)
        with pytest.raises(OutOfDataError):
            wrapper_env.act(_DUMMY_ACTION)

    def test_out_of_data_on_extra_observe(self, wrapper_env):
        while not wrapper_env.is_terminal():
            wrapper_env.act(_DUMMY_ACTION)
        with pytest.raises(OutOfDataError):
            wrapper_env.observe()

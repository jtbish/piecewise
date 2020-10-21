import numpy as np
from piecewise.dtype import DataSpaceBuilder, Dimension

from .gym_environment import GymEnvironment, NormalisedGymEnvironment

_ENV_NAME = "MountainCar-v0"

_POS_LOWER = -1.2
_POS_UPPER = 0.5
_VEL_LOWER = -0.07
_VEL_UPPER = 0.07

_LEFT_ACTION = 0
_RIGHT_ACTION = 2
# exclude "1" action (do nothing)
_CUSTOM_ACTION_SET = {_LEFT_ACTION, _RIGHT_ACTION}


def make_mountain_car_train_env(seed=0,
                                normalise=False,
                                use_default_action_set=False):
    return _make_mountain_car_env(seed,
                                  normalise,
                                  use_default_action_set,
                                  init_obs_type="uniform_both")


def make_mountain_car_test_env(seed=0,
                               normalise=False,
                               use_default_action_set=False):
    return _make_mountain_car_env(seed,
                                  normalise,
                                  use_default_action_set,
                                  init_obs_type="default")


def make_mountain_car_test_env_2(seed=0,
                                 normalise=False,
                                 use_default_action_set=False):
    return _make_mountain_car_env(seed,
                                  normalise,
                                  use_default_action_set,
                                  init_obs_type="uniform_pos_zero_vel")


def _make_mountain_car_env(seed=0,
                           normalise=False,
                           use_default_action_set=False,
                           init_obs_type="default"):
    env = MountainCarEnvironment(seed, use_default_action_set, init_obs_type)
    if normalise:
        return NormalisedGymEnvironment(env)
    else:
        return env


class MountainCarEnvironment(GymEnvironment):
    _VALID_INIT_OBS_TYPES = ("default", "uniform_both", "uniform_pos_zero_vel")

    def __init__(self,
                 seed=0,
                 use_default_action_set=False,
                 init_obs_type="default"):
        custom_obs_space = self._gen_custom_obs_space()
        custom_action_set = \
            self._gen_custom_action_set(use_default_action_set)
        super().__init__(env_name=_ENV_NAME,
                         custom_obs_space=custom_obs_space,
                         custom_action_set=custom_action_set,
                         seed=seed)
        self._rng = np.random.RandomState(seed)
        assert init_obs_type in self._VALID_INIT_OBS_TYPES
        self._init_obs_type = init_obs_type

    def _gen_custom_obs_space(self):
        obs_space_builder = DataSpaceBuilder()
        # order of dims is [pos, vel]
        obs_space_builder.add_dim(Dimension(_POS_LOWER, _POS_UPPER))
        obs_space_builder.add_dim(Dimension(_VEL_LOWER, _VEL_UPPER))
        return obs_space_builder.create_space()

    def _gen_custom_action_set(self, use_default_action_set):
        if use_default_action_set:
            return None
        else:
            return _CUSTOM_ACTION_SET

    def reset(self):
        # call orig reset to let gym reset everything properly in its internals
        obs = super().reset()
        retain_init_obs = (self._init_obs_type == "default")
        if retain_init_obs:
            return obs
        else:
            # create custom starting obs and inject it into the
            # wrapped gym env
            if self._init_obs_type == "uniform_both":
                obs = self._gen_uniform_random_obs()
            elif self._init_obs_type == "uniform_pos_zero_vel":
                obs = self._gen_uniform_pos_zero_vel_obs()
            else:
                assert False
            obs = self._enforce_valid_obs(obs)
            self._wrapped_env.unwrapped.state = obs
            return obs

    def _gen_uniform_random_obs(self):
        obs = []
        for dimension in self._obs_space:
            obs.append(
                self._rng.uniform(low=dimension.lower, high=dimension.upper))
        return np.asarray(obs)

    def _gen_uniform_pos_zero_vel_obs(self):
        pos_dim = self._obs_space[0]
        pos = self._rng.uniform(low=pos_dim.lower, high=pos_dim.upper)
        vel = 0.0
        obs = np.asarray([pos, vel])
        return obs

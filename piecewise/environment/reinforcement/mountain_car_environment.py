from .gym_environment import GymEnvironment, NormalisedGymEnvironment
import numpy as np
from piecewise.dtype import DataSpaceBuilder, Dimension

_ENV_NAME = "MountainCar-v0"

_POS_LOWER = -1.2
_POS_UPPER = 0.5
_VEL_LOWER = -0.07
_VEL_UPPER = 0.07

_LEFT_ACTION = 0
_RIGHT_ACTION = 2
# exclude "1" action (do nothing)
_CUSTOM_ACTION_SET = {_LEFT_ACTION, _RIGHT_ACTION}


def make_mountain_car_train_env(seed=0, normalise=False):
    return _make_mountain_car_env(seed, normalise, modify_init_obss=True)


def make_mountain_car_test_env(seed=0, normalise=False):
    return _make_mountain_car_env(seed, normalise, modify_init_obss=False)


def _make_mountain_car_env(seed=0, normalise=False, modify_init_obss=False):
    env = MountainCarEnvironment(seed, modify_init_obss)
    if normalise:
        return NormalisedGymEnvironment(env)
    else:
        return env


class MountainCarEnvironment(GymEnvironment):
    def __init__(self, seed=0, modify_init_obss=False):
        custom_obs_space = self._gen_custom_obs_space()
        super().__init__(env_name=_ENV_NAME,
                         custom_obs_space=custom_obs_space,
                         custom_action_set=_CUSTOM_ACTION_SET,
                         seed=seed)
        self._rng = np.random.RandomState(seed)
        self._modify_init_obss = modify_init_obss

    def _gen_custom_obs_space(self):
        obs_space_builder = DataSpaceBuilder()
        # order of dims is [pos, vel]
        obs_space_builder.add_dim(Dimension(_POS_LOWER, _POS_UPPER))
        obs_space_builder.add_dim(Dimension(_VEL_LOWER, _VEL_UPPER))
        return obs_space_builder.create_space()

    def reset(self):
        # call orig reset to let gym reset everything properly in its internals
        obs = super().reset()
        if not self._modify_init_obss:
            return obs
        else:
            # create custom starting obs and inject it into the
            # wrapped gym env
            obs = self._gen_uniform_random_obs()
            obs = self._enforce_valid_obs(obs)
            self._wrapped_env.unwrapped.state = obs
            return obs

    def _gen_uniform_random_obs(self):
        obs = []
        for dimension in self._obs_space:
            obs.append(self._rng.uniform(low=dimension.lower,
                high=dimension.upper))
        return np.asarray(obs)

import numpy as np
from piecewise.dtype import DataSpaceBuilder, Dimension
from ..environment import EnvironmentResponse

from .gym_environment import GymEnvironment, NormalisedGymEnvironment

_ENV_NAME = "CartPole-v0"
# These were found via the following procedure:
# Run 1 million trials on raw env only picking left action, recording all
# observations
# Run 1 million trials on raw env only picking right action, recording all
# observations
# From both arrays of observations collected (left and right arrays), calc the
# minimum and maximum values of the cart vel and pole vel features.
# Since ran experiment for so long, the min and max values were almost
# symmetrical around zero.
# Finally take these values and multiply them by a leniency factor of 1.05
_MAX_CART_VEL = 2.3069
_MAX_POLE_VEL = 3.5028
_CART_VEL_LOWER = -(_MAX_CART_VEL)
_CART_VEL_UPPER = _MAX_CART_VEL
_POLE_VEL_LOWER = -(_MAX_POLE_VEL)
_POLE_VEL_UPPER = _MAX_POLE_VEL

_LENIENCY_FACTOR = 1.0
_MAX_CART_POS = 2.4  # from gym env source code
_CART_POS_LOWER = -(_MAX_CART_POS*_LENIENCY_FACTOR)
_CART_POS_UPPER = _MAX_CART_POS*_LENIENCY_FACTOR
_MAX_POLE_ANG_RADIANS = 12*(np.pi/180)  # from gym env source code
_POLE_ANG_LOWER = -(_MAX_POLE_ANG_RADIANS*_LENIENCY_FACTOR)
_POLE_ANG_UPPER = _MAX_POLE_ANG_RADIANS*_LENIENCY_FACTOR


def make_cartpole_train_env(seed=0, normalise=False):
    return _make_cartpole_env(seed, normalise, modify_init_obss=True)


def make_cartpole_test_env(seed=0, normalise=False):
    return _make_cartpole_env(seed, normalise, modify_init_obss=False)


def _make_cartpole_env(seed=0, normalise=False, modify_init_obss=False):
    env = CartpoleEnvironment(seed, modify_init_obss)
    if normalise:
        return NormalisedGymEnvironment(env)
    else:
        return env


class CartpoleEnvironment(GymEnvironment):
    def __init__(self, seed=0, modify_init_obss=False):
        custom_obs_space = self._gen_custom_obs_space()
        super().__init__(env_name=_ENV_NAME,
                         custom_obs_space=custom_obs_space,
                         custom_action_set=None,
                         seed=seed)
        self._rng = np.random.RandomState(seed)
        self._modify_init_obss = modify_init_obss

    def _gen_custom_obs_space(self):
        obs_space_builder = DataSpaceBuilder()
        # order of dims is [cart_pos, cart_vel, pole_ang, pole_vel]
        obs_space_builder.add_dim(Dimension(_CART_POS_LOWER, _CART_POS_UPPER))
        obs_space_builder.add_dim(Dimension(_CART_VEL_LOWER, _CART_VEL_UPPER))
        obs_space_builder.add_dim(Dimension(_POLE_ANG_LOWER, _POLE_ANG_UPPER))
        obs_space_builder.add_dim(Dimension(_POLE_VEL_LOWER, _POLE_VEL_UPPER))
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

    def step(self, action):
        res = super().step(action)
        if res.reward == 1.0:
            reward = 0.01
        else:
            reward = res.reward
        return EnvironmentResponse(obs=res.obs, reward=reward,
                was_correct_action=res.was_correct_action,
                is_terminal=res.is_terminal)

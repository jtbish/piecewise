import gym

from piecewise.dtype import DataSpaceBuilder, Dimension

from .gym_environment import GymEnvironment

_ENV_NAME = "CartPole-v0"
# These were found via the following procedure:
# Run 1 million trials on raw env only picking left action, recording all
# observations
# Run 1 million trials on raw env only picking right action, recording all
# observations
# From both arrays of observations collected (left and right arrays), calc the
# minimum and maximum values of the cart vel and pole val features.
# Since ran experiment for so long, the min and max values were almost
# symmetrical around zero.
# Finally take these values and multiply them by a leniency factor of 1.1
_CART_VEL_LOWER = -2.4167
_CART_VEL_UPPER = 2.4167
_POLE_VEL_LOWER = -3.6696
_POLE_VEL_UPPER = 3.6696


def make_cartpole_environment(seed=0, normalise=False):
    overriden_obs_space = _gen_overriden_obs_space()
    return GymEnvironment(env_name=_ENV_NAME,
                          custom_obs_space=overriden_obs_space,
                          custom_action_set=None,
                          seed=seed,
                          normalise=normalise)


def _gen_overriden_obs_space():
    actual_env = gym.make(_ENV_NAME)
    lower_vector = actual_env.observation_space.low
    upper_vector = actual_env.observation_space.high

    # format of the vectors is [cart_pos, cart_vel, pole_ang, pole_vel]
    # override both vel dims so that they don't span [-inf, inf]
    lower_vector[1] = _CART_VEL_LOWER
    lower_vector[3] = _POLE_VEL_LOWER
    upper_vector[1] = _CART_VEL_UPPER
    upper_vector[3] = _POLE_VEL_UPPER

    obs_space_builder = DataSpaceBuilder()
    for (lower, upper) in zip(lower_vector, upper_vector):
        obs_space_builder.add_dim(Dimension(lower, upper))
    return obs_space_builder.create_space()

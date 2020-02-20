import gym

from piecewise.dtype import DataSpaceBuilder, Dimension

from .gym_environment import GymEnvironment

_ENV_NAME = "CartPole-v0"


def make_cartpole_environment(seed):
    obs_space = _gen_overriden_obs_space()
    return GymEnvironment(env_name=_ENV_NAME,
                          obs_space=obs_space,
                          action_set=None,
                          seed=seed)


def _gen_overriden_obs_space():
    env = gym.make(_ENV_NAME)
    lower_vector = env.observation_space.low
    upper_vector = env.observation_space.high

    # format of the vectors is [cart_pos, cart_vel, pole_ang, pole_vel]
    # override both vel dims so that they don't span [-inf, inf]
    cart_vel_lower = -2.5
    cart_vel_upper = 2.5
    pole_vel_lower = -3.5
    pole_vel_upper = 3.5

    lower_vector[1] = cart_vel_lower
    lower_vector[3] = pole_vel_lower
    upper_vector[1] = cart_vel_upper
    upper_vector[3] = pole_vel_upper

    obs_space_builder = DataSpaceBuilder()
    for (lower, upper) in zip(lower_vector, upper_vector):
        obs_space_builder.add_dim(Dimension(lower, upper))
    return obs_space_builder.create_space()

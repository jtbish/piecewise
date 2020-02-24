import gym

from piecewise.dtype import DataSpaceBuilder, Dimension

from .gym_environment import GymEnvironment


class CartpoleEnvironment(GymEnvironment):
    _ENV_NAME = "CartPole-v0"
    _CART_VEL_LOWER = -2.5
    _CART_VEL_UPPER = 2.5
    _POLE_VEL_LOWER = -3.5
    _POLE_VEL_UPPER = 3.5

    def __init__(self, seed=0):
        obs_space = self._gen_overriden_obs_space()
        super().__init__(env_name=self._ENV_NAME,
                         obs_space=obs_space,
                         action_set=None,
                         seed=seed)

    def _gen_overriden_obs_space(self):
        env = gym.make(self._ENV_NAME)
        lower_vector = env.observation_space.low
        upper_vector = env.observation_space.high

        # format of the vectors is [cart_pos, cart_vel, pole_ang, pole_vel]
        # override both vel dims so that they don't span [-inf, inf]

        lower_vector[1] = self._CART_VEL_LOWER
        lower_vector[3] = self._POLE_VEL_LOWER
        upper_vector[1] = self._CART_VEL_UPPER
        upper_vector[3] = self._POLE_VEL_UPPER

        obs_space_builder = DataSpaceBuilder()
        for (lower, upper) in zip(lower_vector, upper_vector):
            obs_space_builder.add_dim(Dimension(lower, upper))
        return obs_space_builder.create_space()

    def observe(self):
        obs = super().observe()
        self._truncate_obs_velocities(obs)
        return obs

    def _truncate_obs_velocities(self, obs):
        obs[1] = max(obs[1], self._CART_VEL_LOWER)
        obs[1] = min(obs[1], self._CART_VEL_UPPER)

        obs[3] = max(obs[3], self._POLE_VEL_LOWER)
        obs[3] = min(obs[3], self._POLE_VEL_UPPER)

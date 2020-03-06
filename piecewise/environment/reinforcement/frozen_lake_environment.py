import gym
from gym.envs.registration import register


class FrozenLakeGymEnvironment(IEnvironment):
    """Decorator over gym environment containing frozen lake to change the
    observations and observation space to be an (x, y) grid instead of simple
    numbered array of cells."""
    def __init__(self,):
        self._raw_env = GymEnvironment(env_name="FrozenLake8x8-v0", seed=0
        assert isinstance(gym_env, GymEnvironment)
        self._raw_env = gym_env
        self._x_y_grid_obs_space = 
        self._unit_hypercube_obs_space = \
            self._gen_unit_hypercube_obs_space(
                    num_dimensions=len(self._raw_env.obs_space))

from .gym_environment import GymEnvironment

_ENV_NAME = "MountainCar-v0"
_LEFT_ACTION = 0
_RIGHT_ACTION = 2
# exclude "1" action (do nothing)
_CUSTOM_ACTION_SET = {_LEFT_ACTION, _RIGHT_ACTION}


def make_mountain_car_env(seed=0):
    return GymEnvironment(env_name=_ENV_NAME,
                          custom_obs_space=None,
                          custom_action_set=_CUSTOM_ACTION_SET,
                          seed=seed)

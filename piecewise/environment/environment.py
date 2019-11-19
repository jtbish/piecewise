import abc
import functools
from collections import namedtuple

from piecewise.error.environment_error import OutOfDataError


def check_terminal(public_method):
    """Decorator to check if environment is terminal before performing
    operations on it such as observe() and act()."""
    @functools.wraps(public_method)
    def decorator(self, *args, **kwargs):
        if self.is_terminal():
            raise OutOfDataError("Environment is out of data (epoch is "
                                 "finished). Call env.reset() to "
                                 "reinitialise for next epoch.")

        return public_method(self, *args, **kwargs)

    return decorator


EnvironmentResponse = namedtuple("EnvironmentResponse",
                                 ["reward", "was_correct_action"])
CorrectActionNotApplicable = "N/A"


class Environment(metaclass=abc.ABCMeta):
    def __init__(self, obs_space, action_set):
        self._obs_space = obs_space
        self._action_set = action_set
        self.reset()

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_set(self):
        return self._action_set

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def observe(self):
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, action):
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminal(self):
        raise NotImplementedError

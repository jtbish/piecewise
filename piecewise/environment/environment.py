import abc
import functools
from collections import namedtuple
from enum import Enum

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


EnvironmentResponse = namedtuple(
    "EnvironmentResponse", ["reward", "was_correct_action", "is_terminal"])
CorrectActionNotApplicable = "N/A"

EnvironmentStepTypes = Enum("EnivronmentStepTypes",
                            ["single_step", "multi_step"])


class EnvironmentABC(metaclass=abc.ABCMeta):
    """ABC for environments."""
    def __init__(self, obs_space, action_set, step_type):
        self._obs_space = obs_space
        self._action_set = action_set
        self._step_type = step_type
        self.reset()

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_set(self):
        return self._action_set

    @property
    def step_type(self):
        return self._step_type

    @abc.abstractmethod
    def reset(self):
        """Resets the environment to be ready for the next epoch."""
        raise NotImplementedError

    @abc.abstractmethod
    def observe(self):
        """Returns the most recent observation from the environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, action):
        """Performs the given action on the environment, returns an
        EnvironmentResponse named tuple."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminal(self):
        """Checks whether the current epoch of the environment is in a terminal
        state (no more data left to observe)."""
        raise NotImplementedError

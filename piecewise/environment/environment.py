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
    "EnvironmentResponse",
    ["obs", "reward", "was_correct_action", "is_terminal"])
CorrectActionNotApplicable = "N/A"

EnvironmentStepTypes = Enum("EnivronmentStepTypes",
                            ["single_step", "multi_step"])


class IEnvironment(metaclass=abc.ABCMeta):
    """Interface for environments."""
    @property
    @abc.abstractmethod
    def obs_space(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_set(self):
        raise NotImplementedError

    @property
    def step_type(self):
        return self._step_type

    @abc.abstractmethod
    def reset(self):
        """Resets the environment to be ready for the next epoch.

        Returns the initial obs of the epoch."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action):
        """Performs the given action on the environment, returns an
        EnvironmentResponse named tuple."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminal(self):
        """Returns whether the current epoch of the environment is in a terminal
        state."""
        raise NotImplementedError

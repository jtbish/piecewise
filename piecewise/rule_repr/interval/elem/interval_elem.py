import abc
from collections import namedtuple

Interval = namedtuple("Interval", ["lower", "upper"])


class IntervalElem(metaclass=abc.ABCMeta):
    """Element of a condition that represents a 2-tuple."""
    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    def interval(self):
        return Interval(self.lower(), self.upper())

    @abc.abstractmethod
    def lower(self):
        raise NotImplementedError

    @abc.abstractmethod
    def upper(self):
        raise NotImplementedError

    @abc.abstractmethod
    def mutate(self, hyperparams=None):
        raise NotImplementedError

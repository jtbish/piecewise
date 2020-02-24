import abc
from collections import namedtuple

Interval = namedtuple("Interval", ["lower", "upper"])


class IntervalElemABC(metaclass=abc.ABCMeta):
    """ABC for an element of a condition that represents an interval
    predicate."""
    def __init__(self, first_allele, second_allele):
        self._first_allele = first_allele
        self._second_allele = second_allele

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    def interval(self):
        """Returns the full interval spanned by this interval predicate
        in the format [upper, lower]."""
        return Interval(self.lower(), self.upper())

    @abc.abstractmethod
    def lower(self):
        """Returns the lower bound that is represented by the interval
        predicate."""
        raise NotImplementedError

    @abc.abstractmethod
    def upper(self):
        """Returns the upper bound that is represented by the interval
        predicate."""
        raise NotImplementedError

    @abc.abstractmethod
    def mutate(self):
        """Mutates the interval predicate in-place."""
        raise NotImplementedError

    @property
    def first_allele(self):
        return self._first_allele

    @property
    def second_allele(self):
        return self._second_allele

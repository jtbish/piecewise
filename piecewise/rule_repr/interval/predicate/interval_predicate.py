import abc

from ..interval import Interval


class IntervalPredicateABC(metaclass=abc.ABCMeta):
    def __init__(self, first_elem, second_elem):
        self._first_elem = first_elem
        self._second_elem = second_elem

    @abc.abstractmethod
    def lower(self):
        raise NotImplementedError

    @abc.abstractmethod
    def upper(self):
        raise NotImplementedError

    def interval(self):
        return Interval(self.lower(), self.upper())

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{self._first_elem!r}, "
                f"{self._second_elem!r})")

    def __str__(self):
        return f"({self._first_elem}, {self._second_elem})"

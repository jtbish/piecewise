import abc
import functools
import math

from piecewise.error.allele_error import ConversionError

from .config import float_allele_rel_tol
from .formatting import as_truncated_str


class AlleleABC(metaclass=abc.ABCMeta):
    """ABC for alleles.

    An allele is simply a wrapper for a singular value - could be int, float or
    a wildcard."""
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        return f"{self._value}"

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError


def convert_input_to_int(method):
    """Decorator that attempts to convert the single input argument of method
    into an integer."""
    @functools.wraps(method)
    def int_converter(self, input_):
        int_ = _convert_input_to_int(input_)
        return method(self, int_)

    return int_converter


def _convert_input_to_int(input_):
    try:
        int_ = int(input_)
    except ValueError:
        raise ConversionError(f"Cannot convert input '{input_}' to int.")
    return int_


class DiscreteAllele(AlleleABC):
    """Allele operating in discrete (i.e. integer) space."""
    @convert_input_to_int
    def __init__(self, value):
        super().__init__(value)

    def __eq__(self, other):
        if other == DISCRETE_WILDCARD_ALLELE:
            return False
        else:
            other = _convert_input_to_int(other)
            return self._value == other

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"{self._value!r})"

    def __int__(self):
        return self._value


DISCRETE_WILDCARD_ALLELE = "#"


def convert_input_to_float(method):
    """Decorator that attempts to convert the single input argument of method
    to a floating point number."""
    @functools.wraps(method)
    def float_converter(self, input_):
        try:
            float_ = float(input_)
        except ValueError:
            raise ConversionError(f"Cannot convert input '{input_}' to float.")
        return method(self, float_)

    return float_converter


class ContinuousAllele(AlleleABC):
    """Allele operating in continuous (i.e. floating point) space.

    Notably provides an implementation of equality that does a proper floating
    point tolerance check, meaning the syntax allele1 == allele2 can be used by
    clients."""
    @convert_input_to_float
    def __init__(self, value):
        super().__init__(value)

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"{self._value!r})"

    def __str__(self):
        return as_truncated_str(self._value)

    @convert_input_to_float
    def __lt__(self, other):
        return self._value < other

    @convert_input_to_float
    def __le__(self, other):
        return self._value <= other

    @convert_input_to_float
    def __eq__(self, other):
        return math.isclose(self._value, other, rel_tol=float_allele_rel_tol)

    @convert_input_to_float
    def __gt__(self, other):
        return self._value > other

    @convert_input_to_float
    def __ge__(self, other):
        return self._value >= other

    @convert_input_to_float
    def __add__(self, other):
        return type(self)(self._value + other)

    @convert_input_to_float
    def __sub__(self, other):
        return type(self)(self._value - other)

    @convert_input_to_float
    def __iadd__(self, other):
        self._value += other
        return self

    @convert_input_to_float
    def __isub__(self, other):
        self._value -= other
        return self

    def __float__(self):
        return self._value

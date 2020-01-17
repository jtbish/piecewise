import abc
import functools
import math

from piecewise.error.allele_error import ConversionError

from .constants import FLOAT_ALLELE_EQ_REL_TOL


class AlleleABC(metaclass=abc.ABCMeta):
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"{self._value!r})"

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError


def convert_input_to_int(method):
    @functools.wraps(method)
    def int_converter(self, input_):
        try:
            int_ = int(input_)
        except ValueError:
            raise ConversionError(f"Cannot convert input '{input_}' to int.")
        return method(self, int_)

    return int_converter


class IntegerAllele(AlleleABC):
    """Allele operating in discrete (integer) space."""
    @convert_input_to_int
    def __init__(self, value):
        super().__init__(value)

    def __str__(self):
        return f"{self._value}"

    @convert_input_to_int
    def __eq__(self, other):
        return self._value == other

    def __int__(self):
        return self._value


def convert_input_to_float(method):
    @functools.wraps(method)
    def float_converter(self, input_):
        try:
            float_ = float(input_)
        except ValueError:
            raise ConversionError(f"Cannot convert input '{input_}' to float.")
        return method(self, float_)

    return float_converter


class FloatAllele(AlleleABC):
    """Allele operating in continuous (floating point) space."""
    # TODO move this into config file
    _STR_DECIMAL_PLACES = 2

    @convert_input_to_float
    def __init__(self, value):
        super().__init__(value)

    def __str__(self):
        return "{{0:.{0}f}}".format(self._STR_DECIMAL_PLACES).format(
            self._value)

    @convert_input_to_float
    def __lt__(self, other):
        return self._value < other

    @convert_input_to_float
    def __le__(self, other):
        return self._value <= other

    @convert_input_to_float
    def __eq__(self, other):
        return math.isclose(self._value,
                            other,
                            rel_tol=FLOAT_ALLELE_EQ_REL_TOL)

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

import functools
import math

from piecewise.error.classifier_error import AttrUpdateError
from piecewise.lcs.lcs import TIME_STEP_MIN

from .config import classifier_attr_rel_tol
from .formatting import as_truncated_str

TIME_STAMP_MIN = TIME_STEP_MIN
EXPERIENCE_MIN = 0
ACTION_SET_SIZE_MIN = 1
NUMEROSITY_MIN = 1


def check_attr_value(*, min_val, expected_type=None):
    """Decorator to check values given to update classifier attributes.

    Args:
        min_val: The minimum value of the attribute.
        expected_type: The expected type of the attribute, used for checking
            that some attributes are always integers. Default value of None
            means do not check the type of this attribute.
        """
    def decorator(method):
        @functools.wraps(method)
        def _check_attr_value(self, value, *args, **kwargs):
            attr_name = method.__name__
            value_is_large_enough = value >= min_val
            value_is_correct_type = _value_is_correct_type(
                value, expected_type)
            if not (value_is_large_enough and value_is_correct_type):
                raise AttrUpdateError(
                    f"Bad update for {attr_name} attr: "
                    f"value was {value}, expected at least {min_val}, "
                    f"type was {type(value)}, expected {expected_type}")

            return method(self, value, *args, **kwargs)

        return _check_attr_value

    return decorator


def _value_is_correct_type(value, expected_type):
    if expected_type is None:
        # don't check if no type is given
        return True
    else:
        return isinstance(value, expected_type)


class Classifier:
    """A classifier contains a rule (mapping from condition to action), as well as
    other attributes relating to its usage in the system (see properties
    exposed below).

    If a classifier has a numerosity of 1, it is considered to be a
    microclassifier, and if it has a numerosity > 1 it is considered to be a
    macroclassifier. See is_micro and is_macro properties.
    """
    def __init__(self, rule, prediction, error, fitness, time_stamp):
        self._rule = rule
        self._prediction = prediction
        self._error = error
        self._fitness = fitness
        self._time_stamp = time_stamp

        self._experience = EXPERIENCE_MIN
        self._action_set_size = ACTION_SET_SIZE_MIN
        self._numerosity = NUMEROSITY_MIN

    @property
    def rule(self):
        return self._rule

    @rule.setter
    def rule(self, value):
        self._rule = value

    @property
    def condition(self):
        return self._rule.condition

    @property
    def action(self):
        return self._rule.action

    @action.setter
    def action(self, value):
        self._rule.action = value

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, value):
        self._prediction = value

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, value):
        self._error = value

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        self._fitness = value

    @property
    def time_stamp(self):
        return self._time_stamp

    @time_stamp.setter
    @check_attr_value(min_val=TIME_STAMP_MIN, expected_type=int)
    def time_stamp(self, value):
        self._time_stamp = value

    @property
    def experience(self):
        return self._experience

    @experience.setter
    @check_attr_value(min_val=EXPERIENCE_MIN, expected_type=int)
    def experience(self, value):
        self._experience = value

    @property
    def action_set_size(self):
        return self._action_set_size

    @action_set_size.setter
    @check_attr_value(min_val=ACTION_SET_SIZE_MIN)
    def action_set_size(self, value):
        self._action_set_size = value

    @property
    def numerosity(self):
        return self._numerosity

    @numerosity.setter
    @check_attr_value(min_val=NUMEROSITY_MIN, expected_type=int)
    def numerosity(self, value):
        self._numerosity = value

    @property
    def is_micro(self):
        return self._numerosity == NUMEROSITY_MIN

    @property
    def is_macro(self):
        return self._numerosity > NUMEROSITY_MIN

    def generality_as_percentage(self, rule_repr):
        return (self.num_wildcards(rule_repr) / len(self.condition)) * 100

    def num_wildcards(self, rule_repr):
        return rule_repr.num_wildcards(self.condition)

    def __eq__(self, other):
        return self._rule == other.rule and \
            math.isclose(self._prediction, other.prediction,
                         rel_tol=classifier_attr_rel_tol) and \
            math.isclose(self._error, other.error,
                         rel_tol=classifier_attr_rel_tol) and \
            math.isclose(self._fitness, other.fitness,
                         rel_tol=classifier_attr_rel_tol) and \
            self._time_stamp == other.time_stamp and \
            self._experience == other.experience and \
            math.isclose(self._action_set_size, other.action_set_size,
                         rel_tol=classifier_attr_rel_tol) and \
            self._numerosity == other.numerosity

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{self._rule!r}, {self._prediction!r}, "
                f"{self._error!r}, {self._fitness!r}, "
                f"{self._time_stamp!r})")

    def __str__(self):
        return (f"( rule: {self._rule}, "
                f"pred: {as_truncated_str(self._prediction)}, "
                f"err: {as_truncated_str(self._error)}, "
                f"fit: {as_truncated_str(self._fitness)}, "
                f"ts: {self._time_stamp}, "
                f"exp: {self._experience}, "
                f"ass: {as_truncated_str(self._action_set_size)}, "
                f"num: {self._numerosity} )")

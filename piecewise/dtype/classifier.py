import functools
import math

from piecewise.error.classifier_error import AttrUpdateError

TIME_STAMP_MIN = 0
EXPERIENCE_MIN = 0
ACTION_SET_SIZE_MIN = 1
NUMEROSITY_MIN = 1


def check_attr_value(*, min_val):
    def decorator(method):
        @functools.wraps(method)
        def _check_attr_value(self, value, *args, **kwargs):
            if value < min_val:
                attr_name = method.__name__
                raise AttrUpdateError(
                    f"Bad update for {attr_name} attr: must be at least "
                    f"{min_val} but was {value}")
            return method(self, value, *args, **kwargs)

        return _check_attr_value

    return decorator


class Classifier:
    """A classifier contains a rule (mapping from condition to action), as well as
    other attributes relating to its usage in the system (see properties
    exposed below).

    If a classifier has a numerosity of 1, it is considered to be a
    microclassifier, and if it has a numerosity > 1 it is considered to be a
    macroclassifier. See is_micro and is_macro properties.
    """
    _FLOAT_REL_TOL = 1e-5

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
    @check_attr_value(min_val=TIME_STAMP_MIN)
    def time_stamp(self, value):
        self._time_stamp = value

    @property
    def experience(self):
        return self._experience

    @experience.setter
    @check_attr_value(min_val=EXPERIENCE_MIN)
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
    @check_attr_value(min_val=NUMEROSITY_MIN)
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
                         rel_tol=self._FLOAT_REL_TOL) and \
            math.isclose(self._error, other.error,
                         rel_tol=self._FLOAT_REL_TOL) and \
            math.isclose(self._fitness, other.fitness,
                         rel_tol=self._FLOAT_REL_TOL) and \
            self._time_stamp == other.time_stamp and \
            self._experience == other.experience and \
            math.isclose(self._action_set_size, other.action_set_size,
                         rel_tol=self._FLOAT_REL_TOL) and \
            self._numerosity == other.numerosity

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{self._rule!r}, {self._prediction!r}, "
                f"{self._error!r}, {self._fitness!r}, "
                f"{self._time_stamp!r})")

    def __str__(self):
        return (f"( rule: {self._rule}, pred: {self._prediction:.4f}, "
                f"err: {self._error:.4f}, fit: {self._fitness:.4f}, "
                f"ts: {self._time_stamp}, exp: "
                f"{self._experience}, ass: "
                f"{self._action_set_size:.4f}, num: {self._numerosity} )")

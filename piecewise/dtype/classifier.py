import functools
import abc
import math

from piecewise.constants import TIME_STEP_MIN
from piecewise.error.classifier_error import AttrUpdateError

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


class ClassifierABC(metaclass=abc.ABCMeta):
    """ABC for classifiers.

    A classifier contains a rule (mapping from condition to action), as well as
    other attributes relating to its usage in the system (see properties
    exposed below).

    If a classifier has a numerosity of 1, it is considered to be a
    microclassifier, and if it has a numerosity > 1 it is considered to be a
    macroclassifier. See is_micro and is_macro properties.
    """
    def __init__(self, rule, error, fitness, time_stamp):
        self._rule = rule
        self._error = error
        self._niche_min_error = error
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
    def error(self):
        return self._error

    @error.setter
    def error(self, value):
        self._error = value

    @property
    def niche_min_error(self):
        return self._niche_min_error

    @niche_min_error.setter
    def niche_min_error(self, value):
        self._niche_min_error = value

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

    @abc.abstractmethod
    def get_prediction(self, situation=None):
        """Return prediction of the classifier, which may or may not be
        dependent on the situation."""
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class Classifier(ClassifierABC):
    """'Normal' classifier as in canonical XCS - constant prediction."""
    def __init__(self, rule, prediction, error, fitness, time_stamp):
        super().__init__(rule, error, fitness, time_stamp)
        self._prediction = prediction

    def get_prediction(self, situation=None):
        # ignore situation, not needed for constant prediction
        return self._prediction

    def set_prediction(self, value):
        self._prediction = value

    def __eq__(self, other):
        return self._rule == other.rule and \
            math.isclose(self._prediction, other.prediction,
                         rel_tol=classifier_attr_rel_tol) and \
            math.isclose(self._error, other.error,
                         rel_tol=classifier_attr_rel_tol) and \
            math.isclose(self._niche_min_error, other.niche_min_error,
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
                f"nme: {as_truncated_str(self._niche_min_error)}, "
                f"fit: {as_truncated_str(self._fitness)}, "
                f"ts: {self._time_stamp}, "
                f"exp: {self._experience}, "
                f"ass: {as_truncated_str(self._action_set_size)}, "
                f"num: {self._numerosity} )")


class LinearPredictionClassifier(ClassifierABC):
    """Classifier that uses weight vector to compute linear prediction, for
    use with XCSF."""
    _INIT_WEIGHT_VAL = 0.0

    def __init__(self, rule, error, fitness, time_stamp, x_nought):
        super().__init__(rule, error, fitness, time_stamp)
        self._weight_vec = self._init_weight_vec(self._rule.num_features)
        self._x_nought = x_nought

    def _init_weight_vec(self, num_features):
        # weight vec stored as [w_0, w_1, ..., w_n] for n features
        return [self._INIT_WEIGHT_VAL] * (num_features + 1)

    @property
    def weight_vec(self):
        return self._weight_vec

    def get_prediction(self, situation):
        assert len(self._weight_vec) == (len(situation) + 1)
        w_nought = self._weight_vec[0]
        prediction = w_nought * self._x_nought
        for i in range(0, len(situation)):
            s_idx = i
            w_idx = i + 1
            prediction += self._weight_vec[w_idx] * situation[s_idx]
        return prediction

    def __eq__(self, other):
        return self._rule == other.rule and \
            self._weight_vec_is_close(other) and \
            math.isclose(self._error, other.error,
                         rel_tol=classifier_attr_rel_tol) and \
            math.isclose(self._niche_min_error, other.niche_min_error,
                         rel_tol=classifier_attr_rel_tol) and \
            math.isclose(self._fitness, other.fitness,
                         rel_tol=classifier_attr_rel_tol) and \
            self._time_stamp == other.time_stamp and \
            self._experience == other.experience and \
            math.isclose(self._action_set_size, other.action_set_size,
                         rel_tol=classifier_attr_rel_tol) and \
            self._numerosity == other.numerosity

    def _weight_vec_is_close(self, other):
        for (my_elem, other_elem) in zip(self._weight_vec, other._weight_vec):
            if not math.isclose(
                    my_elem, other_elem, rel_tol=classifier_attr_rel_tol):
                return False
        return True

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{self._rule!r}, "
                f"{self._error!r}, {self._fitness!r}, "
                f"{self._time_stamp!r}, {self._x_nought!r})")

    def __str__(self):
        weights = [as_truncated_str(weight) for weight in self._weight_vec]
        return (f"( rule: {self._rule}, "
                f"weights: {weights}), "
                f"err: {as_truncated_str(self._error)}, "
                f"nme: {as_truncated_str(self._niche_min_error)}, "
                f"fit: {as_truncated_str(self._fitness)}, "
                f"ts: {self._time_stamp}, "
                f"exp: {self._experience}, "
                f"ass: {as_truncated_str(self._action_set_size)}, "
                f"num: {self._numerosity} )")

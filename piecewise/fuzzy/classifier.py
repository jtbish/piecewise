import math
import logging

from piecewise.dtype.classifier import (EXPERIENCE_MIN,
                                        Classifier,
                                        LinearPredictionClassifier,
                                        check_attr_value)
from piecewise.dtype.config import classifier_attr_rel_tol
from piecewise.dtype.formatting import as_truncated_str


class FuzzyMixin:
    @property
    def experience(self):
        return self._experience

    @experience.setter
    @check_attr_value(min_val=EXPERIENCE_MIN)
    def experience(self, value):
        """Experience is now a float."""
        self._experience = float(value)

    def calc_matching_degree(self, rule_repr, situation):
        """Calculates matching degree of condition."""
        return rule_repr.eval_condition(self.condition, situation)


class FuzzyClassifier(FuzzyMixin, Classifier):
    def __eq__(self, other):
        return self._rule == other.rule and \
            math.isclose(self._prediction, other.prediction,
                         rel_tol=classifier_attr_rel_tol) and \
            math.isclose(self._error, other.error,
                         rel_tol=classifier_attr_rel_tol) and \
            math.isclose(self._fitness, other.fitness,
                         rel_tol=classifier_attr_rel_tol) and \
            self._time_stamp == other.time_stamp and \
            math.isclose(self._experience, other.experience,
                         rel_tol=classifier_attr_rel_tol) and \
            math.isclose(self._action_set_size, other.action_set_size,
                         rel_tol=classifier_attr_rel_tol) and \
            self._numerosity == other.numerosity

    def __str__(self):
        return (f"( rule: {self._rule}, "
                f"pred: {as_truncated_str(self._prediction)}, "
                f"err: {as_truncated_str(self._error)}, "
                f"fit: {as_truncated_str(self._fitness)}, "
                f"ts: {self._time_stamp}, "
                f"exp: {as_truncated_str(self._experience)}, "
                f"ass: {as_truncated_str(self._action_set_size)}, "
                f"num: {self._numerosity} )")


class FuzzyLinearPredictionClassifier(FuzzyMixin, LinearPredictionClassifier):
    def __eq__(self, other):
        return self._rule == other.rule and \
            self._weight_vec_is_close(other) and \
            math.isclose(self._error, other.error,
                         rel_tol=classifier_attr_rel_tol) and \
            math.isclose(self._fitness, other.fitness,
                         rel_tol=classifier_attr_rel_tol) and \
            self._time_stamp == other.time_stamp and \
            math.is_close(self._experience, other.experience,
                          rel_tol=classifier_attr_rel_tol) and \
            math.isclose(self._action_set_size, other.action_set_size,
                         rel_tol=classifier_attr_rel_tol) and \
            self._numerosity == other.numerosity

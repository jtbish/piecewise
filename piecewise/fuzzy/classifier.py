import math

from piecewise.dtype.classifier import (EXPERIENCE_MIN,
                                        LinearPredictionClassifier,
                                        check_attr_value)
from piecewise.dtype.config import classifier_attr_rel_tol


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

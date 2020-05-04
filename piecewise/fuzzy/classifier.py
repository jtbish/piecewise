from piecewise.dtype.classifier import (EXPERIENCE_MIN,
                                        LinearPredictionClassifier,
                                        check_attr_value)


class FuzzyMixin:
    @property
    def experience(self):
        return self._experience

    @experience.setter
    @check_attr_value(min_val=EXPERIENCE_MIN, expected_type=float)
    def experience(self, value):
        """Experience is now a float."""
        self._experience = value

    @property
    def matching_degree(self):
        """Conditions now have a matching degree."""
        return self._condition.matching_degree


class FuzzyLinearPredictionClassifier(LinearPredictionClassifier, FuzzyMixin):
    pass

from piecewise.dtype import Condition

MIN_MATCHING_DEGREE = 0.0
MAX_MATCHING_DEGREE = 1.0


class FuzzyCondition(Condition):
    """Condition that matches in a fuzzy space, i.e. can partially match.
    Caches its latest matching degree."""
    def __init__(self, genotype):
        super().__init__(genotype)
        self._matching_degree = MIN_MATCHING_DEGREE

    @property
    def matching_degree(self):
        return self._matching_degree

    @matching_degree.setter
    def matching_degree(self, value):
        assert MIN_MATCHING_DEGREE <= value <= MAX_MATCHING_DEGREE
        self._matching_degree = value

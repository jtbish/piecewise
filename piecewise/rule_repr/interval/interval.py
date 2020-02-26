class Interval:
    def __init__(self, lower, upper):
        assert lower <= upper
        self._lower = lower
        self._upper = upper

    def contains_interval(self, other_interval):
        return self._lower <= other_interval._lower and \
                self._upper >= other_interval._upper

    def contains_point(self, point):
        return self._lower <= point <= self._upper

    def fraction_covered_of(self, other_interval):
        # TODO only works for floats
        # trunc bounds of this interval to be compatible with other
        my_lower_trunc = max(self._lower, other_interval._lower)
        my_upper_trunc = min(self._upper, other_interval._upper)

        cover_fraction = (my_upper_trunc - my_lower_trunc) / \
            (other_interval._upper - other_interval._lower)
        assert 0.0 <= cover_fraction <= 1.0
        return cover_fraction

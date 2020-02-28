class ContinuousInterval:
    def __init__(self, lower, upper):
        assert lower <= upper
        self._lower = lower
        self._upper = upper

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    def contains_point(self, point):
        return self._lower <= point <= self._upper

    def contains_interval(self, other_interval):
        """Determines if this interval fully contains the other interval."""
        return self._lower <= other_interval._lower and \
            self._upper >= other_interval._upper

    def fraction_covered_by(self, other_interval):
        """Returns the fraction of *this* interval that is covered *by the
        other interval*.

        e.g. if this interval is [0.0, 1.0] and
        other interval is [0.75, 1.25], result is 0.25, since
        other covers region [0.75, 1.0] in this.
        """
        # trunc bounds of other interval to be compatible with this interval
        # (so that other interval falls within this interval)
        other_lower_trunc = max(other_interval.lower, self._lower)
        other_upper_trunc = min(other_interval._upper, self._upper)
        cover_fraction = (other_upper_trunc - other_lower_trunc) / \
            (self._upper - self._lower)
        assert 0.0 <= cover_fraction <= 1.0
        return cover_fraction

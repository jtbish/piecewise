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

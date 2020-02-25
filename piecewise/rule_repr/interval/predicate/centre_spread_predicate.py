from .interval_predicate import IntervalPredicateABC


class CentreSpreadPredicate(IntervalPredicateABC):
    """Represents a (centre, spread) predicate."""
    def __init__(self, centre, spread):
        super().__init__(first_elem=centre, second_elem=spread)
        self._centre = self._first_elem
        self._spread = self._second_elem

    def lower(self):
        return self._centre - self._spread

    def upper(self):
        return self._centre + self._spread

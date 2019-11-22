from collections import defaultdict


class PopulationState:
    """Keeps track of changes made to the state of a Population instance.

    Modifications are measured in units of *number of microclassifiers*.
    """
    def __init__(self):
        self._state_changes = defaultdict(lambda: 0)

    def query(self):
        return self._state_changes

    def update(self, track_label, micros_delta):
        if track_label is not None:
            self._state_changes[track_label] += micros_delta

    def as_dict(self):
        return dict(self._state_changes)

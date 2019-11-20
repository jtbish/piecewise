from collections import defaultdict


class PopulationState:
    """Keeps track of changes made to the state of a Population instance.

    Modifications are measured in units of *number of microclassifiers*.
    E.g. if the state has recorded that five deletions have been performed,
    then five microclassifiers have been deleted from the population since its
    creation."""
    def __init__(self):
        self._state_changes = defaultdict(lambda: 0)

    def query(self):
        return self._state_changes

    def update(self, track_label, change):
        if track_label is not None:
            self._state_changes[track_label] += change

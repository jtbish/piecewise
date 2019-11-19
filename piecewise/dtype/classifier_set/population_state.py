from collections import defaultdict


class PopulationState:
    """Keeps track of modifications made to the state of a Population instance.

    Modifications are measured in units of *number of microclassifiers*.
    E.g. if the state has recorded that five deletions have been performed,
    then five microclassifiers have been deleted from the population since its
    creation."""
    def __init__(self):
        self._state_modifications = defaultdict(lambda: 0)

    def query(self):
        return self._state_modifications

    def update(self, tracker_name, increment):
        if tracker_name is not None:
            self._state_modifications[tracker_name] += increment

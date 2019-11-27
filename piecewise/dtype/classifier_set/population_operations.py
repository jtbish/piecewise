from collections import defaultdict


class PopulationOperations:
    """Stores changes made to the state of a Population instance via
    its tracked operations.

    Modifications are measured in units of *number of microclassifiers*.
    """
    def __init__(self):
        self._operations_performed = defaultdict(lambda: 0)

    def query(self):
        return dict(self._operations_performed)

    def update(self, track_label, micros_delta):
        if track_label is not None:
            self._operations_performed[track_label] += micros_delta

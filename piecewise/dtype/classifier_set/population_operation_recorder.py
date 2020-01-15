from collections import UserDict


class PopulationOperationRecorder(UserDict):
    """Records counts for atomic operations performed on a Population instance.

    Each operation type recorded acts at the level of abstraction of an
    individual microclassifier. Thus the counts can be interpreted in units of
    "number of microclassifiers".
    """
    def __getitem__(self, key):
        operation_label = key
        try:
            return self.data[operation_label]
        except KeyError:
            return 0

    def __setitem__(self, key, value):
        operation_label = key
        self.data[operation_label] = value

class Condition:
    """Mutable sequence type that represents a condition of a rule.

    Basically just a stripped-down version of list that supports a limited
    subset of operations.
    """
    def __init__(self, elems=None):
        if elems is None:
            self._elems = []
        else:
            self._elems = list(elems)

    @classmethod
    def from_elem_args(cls, *elems):
        return cls(elems)

    def append(self, elem):
        self._elems.append(elem)

    def __eq__(self, other):
        for (my_elem, other_elem) in zip(self._elems, other._elems):
            if my_elem != other_elem:
                return False
        return True

    def __setitem__(self, idx, value):
        self._elems[idx] = value

    def __getitem__(self, idx):
        return self._elems[idx]

    def __len__(self):
        return len(self._elems)

    def __iter__(self):
        return iter(self._elems)

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"{self._elems!r})"

    def __str__(self):
        return "(" + ", ".join([str(elem) for elem in self._elems]) + ")"

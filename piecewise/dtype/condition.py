class DiscreteCondition:
    """Mutable sequence type that represents a condition of a discrete rule
    representation.

    Basically just a stripped-down version of list that supports a limited
    subset of operations.
    """
    def __init__(self, elems):
        self._elems = list(elems)

    @classmethod
    def from_elem_args(cls, *elems):
        """Create a condition from an element arguments."""
        return cls(elems)

    def append(self, elem):
        """Append the given element to the end of the condition."""
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


class IntervalCondition:
    """Mutable sequence type that represents a condition of an interval-based
    rule representation.

    The reason this data structure exists is because operations on conditions
    with intervals sometimes need to view the condition as a sequence of
    alleles, and sometimes as a sequence of interval predicates.

    Of these two views, the allele is more "raw", as interval predicates are
    composed of two alleles. Thus, the philosophy of this data sturcutre is to
    store the data that comprises the condition internally as a sequence of
    alleles, and *if* a caller requests that the data wants to view the data as
    interval predicates, these are constructed on the fly.

    Callers who want to view the alleles are just given access to the internal
    allele sequence. This way, both views are available and any mutations done
    by callers are acting on the shared, only, representation of the data; the
    allele sequence."""
    def __init__(self, elems):
        self._alleles = self._convert_elems_to_alleles(elems)
        self._elem_type = type(elems[0])

    def _convert_elems_to_alleles(self, elems):
        alleles = []
        for elem in elems:
            alleles.append(elem.first_allele)
            alleles.append(elem.second_allele)
        return alleles

    @classmethod
    def from_elem_args(cls, *elems):
        return cls(elems)

    @property
    def alleles(self):
        return list(self._alleles)

    @property
    def elems(self):
        return self._construct_elems_from_alleles()

    def _construct_elems_from_alleles(self):
        assert len(self._alleles) % 2 == 0
        elems = []
        for first_allele_idx in range(0, len(self._alleles), 2):
            elem_first_allele = self._alleles[first_allele_idx]
            elem_second_allele = self._alleles[first_allele_idx + 1]
            elems.append(self._elem_type(elem_first_allele,
                                         elem_second_allele))
        return elems

    def __eq__(self, other):
        for (my_allele, other_allele) in zip(self._alleles, other._alleles):
            if my_allele != other_allele:
                return False
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"{self.elems!r})"

    def __str__(self):
        return "(" + ", ".join([str(elem) for elem in self.elems]) + ")"

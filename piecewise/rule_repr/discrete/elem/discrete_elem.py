WILDCARD_CHAR = "#"


class DiscreteElem:
    def __init__(self, allele):
        self._allele = allele

    def __eq__(self, other):
        try:
            return self._allele == other._allele
        except AttributeError:
            if type(other) == DiscreteWildcardElem:
                # forward to DiscreteWildcardElem __eq__
                return NotImplemented
            else:
                # let allele deal with other
                return self._allele == other

    def __repr__(self):
        return f"{self.__class__.__name__}({self._allele!r})"

    def __str__(self):
        return f"{self._allele}"


class DiscreteWildcardElem:
    def __eq__(self, other):
        return type(other) == type(self)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return WILDCARD_CHAR

class DataSpaceBuilder:
    """Convenience class to make syntax of building data space nicer."""
    def __init__(self):
        self._dims = []

    def add_dim(self, dimension):
        self._dims.append(dimension)

    def create_space(self):
        return tuple(self._dims)

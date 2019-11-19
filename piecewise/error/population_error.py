from .core_errors import PiecewiseError


class PopulationError(PiecewiseError):
    pass


class InvalidSizeError(PopulationError):
    """Indicates population was instantiated with an invalid capacity."""
    pass

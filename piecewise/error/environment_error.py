from .core_errors import PiecewiseError


class EnvError(PiecewiseError):
    pass


class OutOfDataError(EnvError):
    """Indicates that Environment instance has cycled through all available data
    and can no longer be iterated."""
    pass


class InvalidSpecError(EnvError):
    """Indicates specification of an environment instance was invalid."""
    pass

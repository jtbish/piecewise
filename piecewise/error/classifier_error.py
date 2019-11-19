from .core_errors import PiecewiseError


class ClassifierError(PiecewiseError):
    pass


class AttrUpdateError(ClassifierError):
    """Indicates problem with updating property of a classifier."""
    pass

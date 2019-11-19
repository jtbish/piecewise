from .core_errors import PiecewiseError


class ClassifierSetError(PiecewiseError):
    pass


class MemberNotFoundError(ClassifierSetError):
    """Indicates a specified member was not found in a classifier set."""
    pass

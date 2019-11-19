class PiecewiseError(Exception):
    """Base error class for framework."""
    def __init__(self, message=""):
        self._message = message

    def __str__(self):
        return self._message


class InternalError(PiecewiseError):
    """Error indicating internal bug or logical flaw, not meant to be raised
    under normal operation."""
    pass

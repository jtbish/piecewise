from piecewise.error.core_errors import PiecewiseError


class AlgorithmError(PiecewiseError):
    pass


class InvalidSpecError(AlgorithmError):
    pass

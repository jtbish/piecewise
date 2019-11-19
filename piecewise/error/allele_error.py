from .core_errors import PiecewiseError


class AlleleError(PiecewiseError):
    pass


class ConversionError(AlleleError):
    """Indicates bad input was given to allele constructor or magic method."""
    pass

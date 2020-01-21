from .config import str_decimal_places


def as_truncated_str(float_):
    """Returns the given float as string with the user-defined number of
    decimal places."""
    return "{{0:.{0}f}}".format(str_decimal_places).format(float_)

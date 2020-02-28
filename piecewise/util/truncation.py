def truncate_val(val, lower_bound, upper_bound):
    """Truncate val to be in the range [lower_bound, upper_bound]."""
    val = max(val, lower_bound)
    val = min(val, upper_bound)
    return val

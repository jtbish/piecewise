class LinguisticVar:
    """Linguistic var has underlying fuzzy sets / membership funcs associated
    with it."""
    def __init__(self, membership_funcs, name):
        self._membership_funcs = tuple(membership_funcs)
        self._name = name

    @property
    def membership_funcs(self):
        return self._membership_funcs

    @property
    def name(self):
        return self._name

    def eval_membership_funcs(self, membership_func_idxs, input_scalar):
        result = []
        to_eval = [self._membership_funcs[idx] for idx in membership_func_idxs]
        for membership_func in to_eval:
            result.append(membership_func.fuzzify(input_scalar))
        return tuple(result)

    def eval_all_membership_funcs(self, input_scalar):
        result = []
        for membership_func in self._membership_funcs:
            result.append(membership_func.fuzzify(input_scalar))
        return tuple(result)

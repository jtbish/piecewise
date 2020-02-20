class _HyperparamsRegistry:
    def __init__(self):
        self._dict = {}

    def register(self, dict_):
        for name, value in dict_.items():
            self._dict[name] = value

    def __getitem__(self, key):
        name = key
        return self._dict[name]

    def __setitem__(self, key, value):
        raise NotImplementedError


_hyperparams_registry = _HyperparamsRegistry()

# mask access to registry behind accessor functions to ensure callers can't
# mutate it by accident - it is "immutable"


def register_hyperparams(hyperparams_dict):
    _hyperparams_registry.register(hyperparams_dict)


def get_hyperparam(name):
    return _hyperparams_registry[name]

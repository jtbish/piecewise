class HyperparamsRegistry:
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


hyperparams_registry = HyperparamsRegistry()

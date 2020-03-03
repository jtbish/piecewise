_hyperparams_registry = {}

# mask access to registry behind accessor functions to ensure callers can't
# mutate it by accident - it is "immutable"


def register_hyperparams(hyperparams_dict):
    global _hyperparams_registry
    _hyperparams_registry = {**_hyperparams_registry, **hyperparams_dict}


def get_hyperparam(name):
    return _hyperparams_registry[name]


def get_registry():
    return _hyperparams_registry

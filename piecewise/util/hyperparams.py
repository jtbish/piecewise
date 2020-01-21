from collections import defaultdict


class ParamRegistry:
    """Utility class to hold a collection of parameters for different "users".

    The main use case for this is building a collection of hyperparams for an
    algorithm when some of the hyperparams are shared by different components,
    as is the case for something like XCS where e.g. the beta hyperparam
    is shared by both the fitness update and credit assignment components.

    Why is this even needed? Well, in the previous case, it allows the client
    code to enter the value for a hyperparam *once*, but associate it with
    multiple "users".

    The concept of a "user" is deliberately abstract but in the example of XCS
    it is either the root algorithm logic or one of its components.

    After all parameters have been registered then specific parameters for a
    user can be neatly extracted into a separate dict - useful for e.g.
    ensuring that algorithm components only know about their own hyperparams.

    Example usage:

    >>> params = ParamRegistry()
    >>> params.register("alpha", 0.1, used_by=("user1", "user2"))
    >>> params.register("beta", 0.2, used_by=("user1", "user3"))
    >>> params.for_user("user1")
    {'alpha': 0.1, 'beta': 0.2}
    >>> params.for_user("user2")
    {'alpha': 0.1}
    >>> params.for_user("user3")
    {'beta': 0.2}
    >>> params.for_user("user4")
    {}
    """
    def __init__(self):
        self._dict = defaultdict(lambda: {})

    def register(self, param_name, param_value, *, used_by):
        """used_by is an iterable of user names."""
        for user_name in used_by:
            self._merge_into_user_params(user_name, param_name, param_value)

    def _merge_into_user_params(self, user_name, param_name, param_value):
        existing_params = self._dict[user_name]
        updated_params = {**existing_params, **{param_name: param_value}}
        self._dict[user_name] = updated_params

    def for_user(self, user_name):
        return self._dict[user_name]

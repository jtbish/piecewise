from collections import defaultdict


class Hyperparams:
    """Utility class to hold a collection of hyperparams for different "users".

    The main use case for this is building a collection of hyperparams for an
    algorithm when some of the hyperparams are shared by different components,
    as is the case for something like XCS where e.g. the beta hyperparam
    is shared by both the fitness update and credit assignment components.

    Why is this even needed? Well, in the previous case, it allows the client
    code to enter the value for a hyperparam *once*, but associate it with
    possibly many "users".

    The concept of a "user" is deliberately abstract but in the example of XCS
    it is either the root algorithm itself or one of its components.

    After all hyperparams have been registered then the specific hyperparams
    for a user can be neatly extracted into a separate dict - useful for
    ensuring that algorithm components only know about their own hyperparams.

    Example usage:

    >>> hyperparams = Hyperparams()
    >>> hyperparams.register("alpha", 0.1, used_by=("user1", "user2"))
    >>> hyperparams.register("beta", 0.2, used_by=("user1", "user3"))
    >>> hyperparams.get("user1")
    {'alpha': 0.1, 'beta': 0.2}
    >>> hyperparams.get("user2")
    {'alpha': 0.1}
    >>> hyperparams.get("user3")
    {'beta': 0.2}
    >>> hyperparams.get("user4")
    {}
    """
    def __init__(self):
        self._dict = defaultdict(lambda: {})

    def register(self, hyperparam_name, hyperparam_value, *, used_by):
        for user_name in used_by:
            self._merge_into_user_hyperparams(user_name, hyperparam_name,
                                              hyperparam_value)

    def _merge_into_user_hyperparams(self, user_name, hyperparam_name,
                                     hyperparam_value):
        existing_hyperparams = self._dict[user_name]
        updated_hyperparams = {
            **existing_hyperparams,
            **{
                hyperparam_name: hyperparam_value
            }
        }
        self._dict[user_name] = updated_hyperparams

    def get(self, user_name):
        return self._dict[user_name]

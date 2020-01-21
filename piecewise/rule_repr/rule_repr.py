import abc


class IRuleRepr(metaclass=abc.ABCMeta):
    """Interface for rule representations."""
    @abc.abstractmethod
    def does_match(self, condition, situation):
        """Determines if a condition matches a situation."""
        raise NotImplementedError

    @abc.abstractmethod
    def gen_covering_condition(self, situation, hyperparams=None):
        """Generates and returns a covering condition for the given
        situation.

        May or may not require hyperparameters, hence the
        hyperparams argument is None by default."""
        raise NotImplementedError

    @abc.abstractmethod
    def mutate_condition(self, condition, hyperparams=None, situation=None):
        """Mutates the given condition in-place.

        May or may not require hyperparameters or an environmental situation,
        hence these arguments are None by default."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_wildcard(self, condition_elem, elem_idx=None):
        """Determines if the given element of a condition (along with its index
        in the condition) is (equivalent) to a wildcard.

        May or may not require the index of the element in its condition,
        hence the elem_idx argument is None by default."""
        raise NotImplementedError

    @abc.abstractmethod
    def num_wildcards(self, condition):
        """Returns the number of wildcard elements in the condition."""
        raise NotImplementedError

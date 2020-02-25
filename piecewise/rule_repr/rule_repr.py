import abc


class IRuleRepr(metaclass=abc.ABCMeta):
    """Interface for rule representations."""
    @abc.abstractmethod
    def does_match(self, condition, situation):
        """Determines if a condition matches a situation."""
        raise NotImplementedError

    @abc.abstractmethod
    def gen_covering_condition(self, situation):
        """Generates and returns a covering condition for the given
        situation."""
        raise NotImplementedError

    @abc.abstractmethod
    def crossover_conditions(self, first_condition, second_condition,
                             crossover_strat):
        """Crosses over the given conditions in place, using the specified
        crossover strategy."""
        raise NotImplementedError

    @abc.abstractmethod
    def mutate_condition(self, condition, situation=None):
        """Mutates the given condition in-place.

        May or may not require hyperparameters or an environmental situation,
        hence these arguments are None by default."""
        raise NotImplementedError

    @abc.abstractmethod
    def calc_generality(self, condition):
        """Returns the generality of the condition as a fraction,
        i.e. how much of the input space the condition covers."""
        raise NotImplementedError

    @abc.abstractmethod
    def genotype_to_phenotype(self, condition):
        """Converts the given condition (in genotype space) to its
        representation in phenotype space."""
        raise NotImplementedError

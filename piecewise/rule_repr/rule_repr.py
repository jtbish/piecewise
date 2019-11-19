import abc


class RuleRepr(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def does_match(self, condition, situation):
        raise NotImplementedError

    @abc.abstractmethod
    def gen_covering_condition(self, situation, hyperparams):
        raise NotImplementedError

    @abc.abstractmethod
    def mutate_condition(self, condition, hyperparams, situation=None):
        raise NotImplementedError

    @abc.abstractmethod
    def is_wildcard(self, condition_elem, elem_idx):
        raise NotImplementedError

    @abc.abstractmethod
    def num_wildcards(self, condition):
        raise NotImplementedError

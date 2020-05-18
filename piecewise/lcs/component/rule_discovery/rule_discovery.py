import abc


class IRuleDiscoveryStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, action_set, population, situation, time_step):
        raise NotImplementedError


class NullRuleDiscovery(IRuleDiscoveryStrategy):
    def __call__(self, action_set, population, situation, time_step):
        pass

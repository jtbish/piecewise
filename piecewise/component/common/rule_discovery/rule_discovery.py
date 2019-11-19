import abc


class RuleDiscoveryStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self,
                 env_action_set=None,
                 subsumption_strat=None,
                 hyperparams=None):
        self._env_action_set = env_action_set
        self._subsumption_strat = subsumption_strat
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def __call__(self, operating_set, population, situation, time_step):
        """Discovers new classifiers and inserts them into the population."""
        raise NotImplementedError

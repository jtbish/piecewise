import abc

from ..environment import Environment, EnvironmentStepTypes


class AbstractReinforcementEnvironment(Environment, metaclass=abc.ABCMeta):
    def __init__(self, obs_space, action_set):
        step_type = EnvironmentStepTypes.multi_step
        super().__init__(obs_space, action_set, step_type)

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def observe(self):
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, action):
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminal(self):
        raise NotImplementedError

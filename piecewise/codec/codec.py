import abc


class Codec(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def encode(self, obs):
        """Takes observation from environment and encodes it, returning
        situation for internal use in algorithm."""
        raise NotImplementedError

    @abc.abstractmethod
    def make_situation_space(self, obs_space):
        raise NotImplementedError


class NullCodec(Codec):
    def encode(self, obs):
        return obs

    def make_situation_space(self, obs_space):
        return obs_space

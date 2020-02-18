import abc
import copy
import pickle

from piecewise.util import ParametrizedMixin


class MonitorABC(ParametrizedMixin, metaclass=abc.ABCMeta):
    def __init__(self):
        # keys of history are time steps of updates
        self._history = {}

    @abc.abstractmethod
    def update(self, lcs):
        raise NotImplementedError

    def query(self):
        return self._history

    def save(self, save_path):
        filename = f"{self.__class__.__name__}.pkl"
        with open(save_path / filename, "wb") as fp:
            pickle.dump(self.query(), fp)


# TODO move deepcopies below into LCS getters?


class PopulationMonitor(MonitorABC):
    def __init__(self, update_freq):
        assert update_freq > 0
        self._update_freq = update_freq
        super().__init__()

        self.record_parametrization(update_freq=update_freq)

    def update(self, lcs):
        if self._should_update(lcs.time_step):
            self._update(lcs)

    def _should_update(self, time_step):
        return time_step % self._update_freq == 0

    def _update(self, lcs):
        pop_copy = copy.deepcopy(lcs.population)
        self._history[lcs.time_step] = pop_copy


class LoopMonitor(MonitorABC):
    def update(self, lcs):
        loop_data_copy = copy.deepcopy(lcs.loop_data)
        self._history[lcs.time_step] = loop_data_copy

import abc
import copy
import pickle


class MonitorABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, time_step, value):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, save_path):
        raise NotImplementedError


class Monitor(MonitorABC):
    def __init__(self, name, update_freq=1):
        assert update_freq > 0
        self._name = name
        self._update_freq = update_freq
        self._history = {}

    def update(self, time_step, value):
        if self._should_update(time_step):
            # take a deepcopy of the value to make sure the refs stored in
            # self._history over time are different
            self._history[time_step] = copy.deepcopy(value)

    def _should_update(self, time_step):
        return time_step % self._update_freq == 0

    def save(self, save_path):
        filename = f"{self._name}_monitor.pkl"
        with open(save_path / filename, "wb") as fp:
            pickle.dump(self._history, fp)


class NullMonitor(MonitorABC):
    def update(self, time_step, value):
        pass

    def save(self, save_path):
        pass

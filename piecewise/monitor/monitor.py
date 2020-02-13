import abc
import logging
from collections import namedtuple

MonitorItem = namedtuple("MonitorItem", ["name", "callback_func"])


class IMonitor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def try_update(self, experiment):
        raise NotImplementedError

    @abc.abstractmethod
    def query(self):
        raise NotImplementedError


class Monitor(IMonitor):
    def __init__(self, items, update_freq):
        """Update freq in units of 'number of epochs'."""
        self._items = items
        self._items_history = {item.name: [] for item in self._items}
        self._update_freq = update_freq

    def try_update(self, experiment):
        logging.debug("Trying to update monitor.")
        if self._should_update(experiment.time_step):
            logging.debug("Updating monitor.")
            self._update(experiment)

    def _should_update(self, epoch_num):
        return epoch_num % self._update_freq == 0

    def _update(self, experiment):
        for item in self._items:
            self._update_item_history(experiment, item)

    def _update_item_history(self, experiment, item):
        item_history = self._items_history[item.name]
        item_history.append(item.callback_func(experiment))

    def query(self):
        return self._items_history


class NullMonitor(IMonitor):
    def try_update(self, experiment):
        pass

    def query(self):
        pass

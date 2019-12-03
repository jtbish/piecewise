import abc
from collections import namedtuple

MonitorItem = namedtuple("MonitorItem", ["name", "callback_func"])


class AbstractMonitor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, lcs):
        raise NotImplementedError

    @abc.abstractmethod
    def query(self):
        raise NotImplementedError


class NullMonitor(AbstractMonitor):
    def update(self, lcs):
        pass

    def query(self):
        pass


class Monitor(AbstractMonitor):
    def __init__(self, *items):
        self._items = items
        self._items_history = {item.name: [] for item in self._items}

    def update(self, lcs):
        for item in self._items:
            self._update_item_history(lcs, item)

    def _update_item_history(self, lcs, item):
        item_history = self._items_history[item.name]
        item_history.append(item.callback_func(lcs))

    def query(self):
        return self._items_history

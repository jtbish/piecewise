from collections import namedtuple

MonitorItem = namedtuple("MonitorItem", ["name", "callback_func"])


class Monitor:
    def __init__(self, items):
        self._items = items
        self._items_history = {item.name: [] for item in self._items}

    def update(self, experiment):
        for item in self._items:
            self._update_item_history(experiment, item)

    def _update_item_history(self, experiment, item):
        item_history = self._items_history[item.name]
        item_history.append(item.callback_func(experiment))

    def query(self):
        return self._items_history

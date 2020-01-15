import abc


class AbstractMonitor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, lcs, population, epoch_num):
        raise NotImplementedError

    @abc.abstractmethod
    def query(self, sub_monitor_cls):
        raise NotImplementedError


class NullMonitor(AbstractMonitor):
    def update(self, lcs, epoch_num):
        pass

    def query(self, sub_monitor_cls):
        pass


class Monitor(AbstractMonitor):
    def __init__(self, *sub_monitors):
        self._sub_monitors = {}
        for sub_monitor in sub_monitors:
            sub_monitor_cls_name = type(sub_monitor).__name__
            self._sub_monitors[sub_monitor_cls_name] = sub_monitor

    def update(self, lcs, population, epoch_num):
        for sub_monitor in self._sub_monitors.values():
            sub_monitor.update(lcs, population, epoch_num)

    def query(self, sub_monitor_cls):
        sub_monitor_to_query = self._sub_monitors[sub_monitor_cls.__name__]
        return sub_monitor_to_query.query()

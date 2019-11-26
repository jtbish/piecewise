import abc


class AbstractMonitor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, lcs, epoch_num):
        raise NotImplementedError

    @abc.abstractmethod
    def report(self):
        raise NotImplementedError


class NullMonitor(AbstractMonitor):
    def update(self, lcs, epoch_num):
        pass

    def report(self):
        pass


class Monitor(AbstractMonitor):
    def __init__(self, *sub_monitors):
        self._sub_monitors = sub_monitors

    def update(self, lcs, epoch_num):
        for sub_monitor in self._sub_monitors:
            sub_monitor.update(lcs, epoch_num)

    def report(self):
        for sub_monitor in self._sub_monitors:
            sub_monitor.report()

import abc


class AbstractMonitor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, lcs):
        raise NotImplementedError

    @abc.abstractmethod
    def report(self, epoch_num):
        raise NotImplementedError

    @abc.abstractmethod
    def plot(self):
        raise NotImplementedError


class NullMonitor(AbstractMonitor):
    def update(self, lcs):
        pass

    def report(self, epoch_num):
        pass

    def plot(self):
        pass


class Monitor(AbstractMonitor):
    def __init__(self, sub_monitors):
        self._sub_monitors = sub_monitors

    @classmethod
    def from_sub_monitor_classes(cls, *sub_monitor_classes):
        return cls(
            [sub_monitor_cls() for sub_monitor_cls in sub_monitor_classes])

    def update(self, lcs):
        for sub_monitor in self._sub_monitors:
            sub_monitor.update(lcs)

    def report(self, epoch_num):
        print(f"Epoch {epoch_num}:")
        for sub_monitor in self._sub_monitors:
            sub_monitor.report()
        print("\n")

    def plot(self):
        for sub_monitor in self._sub_monitors:
            sub_monitor.plot()

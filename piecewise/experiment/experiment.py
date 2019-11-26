from piecewise.monitor import NullMonitor


class Experiment:
    """High-level interface to LCS class."""
    def __init__(self, lcs, monitor=None, num_epochs=1):
        self._lcs = lcs
        if monitor is None:
            self._monitor = NullMonitor()
        else:
            self._monitor = monitor
        self._num_epochs = num_epochs

    def run(self):
        epoch_num = 1
        while epoch_num < self._num_epochs:
            population = self._lcs.train_single_epoch()
            self._monitor.update(self._lcs, epoch_num)
            epoch_num += 1
        return population, self._monitor

    def archive(self):
        raise NotImplementedError

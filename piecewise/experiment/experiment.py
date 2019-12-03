import random

import numpy as np

from piecewise.monitor import NullMonitor


class Experiment:
    def __init__(self, lcs, seed, num_epochs=1, monitor=None):
        self._lcs = lcs
        self._seed = seed
        self._seed_rngs(self._seed)
        self._num_epochs = num_epochs
        self._monitor = self._init_monitor(monitor)

    @property
    def seed(self):
        return self._seed

    def _seed_rngs(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _init_monitor(self, monitor):
        if monitor is None:
            monitor = NullMonitor()
        return monitor

    def run(self):
        for epoch_num in range(1, self._num_epochs + 1):
            self._lcs.train_single_epoch()
            self._monitor.update(self._lcs)
        return self._lcs.population, self._monitor.query()

    def archive(self):
        raise NotImplementedError

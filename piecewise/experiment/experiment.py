import pickle
from pathlib import Path

from piecewise.monitor import NullMonitor


class Experiment:
    def __init__(self, env, alg, num_training_samples, logging="verbose",
            archive_path=None):
        self._runner = ExperimentRunner()
        self._logger = ExperimentLogger()
        self._archiver = ExperimentArchiver()

    def __init__(self, tag, lcs, num_epochs=1, monitor=None):
        self._tag = tag
        self._lcs = lcs
        self._num_epochs = num_epochs
        self._monitor = self._init_monitor(monitor)

    def _init_monitor(self, monitor):
        if monitor is None:
            monitor = NullMonitor()
        return monitor

    def run(self):
        self._runner.run()
#        for epoch_num in range(self._num_epochs):
#            self._lcs.train_single_epoch()
#            self._monitor.update(self._lcs)
#        self._monitor_output = self._monitor.query()
#        return self._lcs.population, self._monitor_output

    def archive(self):
        self._archiver.archive()
#        archive_path = self._make_archive_dir()
#        self._pickle_population(archive_path)
#        self._pickle_monitor_output(archive_path)

    def _make_archive_dir(self):
        archive_path = Path(f"./{self._tag}_archive")
        archive_path.mkdir(exist_ok=True)
        return archive_path

    def _pickle_population(self, archive_path):
        with open(archive_path / "pop.pkl", "wb") as fp:
            pickle.dump(self._lcs.population, fp)

    def _pickle_monitor_output(self, archive_path):
        with open(archive_path / "monitor_out.pkl", "wb") as fp:
            pickle.dump(self._monitor_output, fp)

def ExperimentRunner:
    def __init__(self):
        pass

def ExperimentLogger:
    def __init__(self):
        pass

def ExperimentArchiver:
    def __init__(self):
        pass

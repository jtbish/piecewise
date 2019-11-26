from piecewise.monitor import NullMetaMonitor


class MetaExperiment:
    def __init__(self, experiments, meta_monitor=None):
        self._experiments = experiments
        if meta_monitor is None:
            self._meta_monitor = NullMetaMonitor()
        else:
            self._meta_monitor = meta_monitor

    def run_experiments(self):
        collected_monitors = []
        for experiment in self._experiments:
            population, monitor = experiment.run()
            collected_monitors.append(monitor)
        self._meta_monitor.process(collected_monitors)

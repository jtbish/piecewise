import logging
import shutil
from pathlib import Path

import __main__
from piecewise.error.experiment_error import ExperimentError
from piecewise.lcs import LCS


class Experiment:
    def __init__(self,
                 save_dir,
                 env,
                 alg,
                 num_training_samples,
                 monitors=None,
                 logging_level=logging.INFO):
        self._monitors = self._init_monitors(monitors)
        self._lcs = LCS(env, alg, num_training_samples)
        self._setup_logging(logging_level)
        self._save_path = self._setup_save_path(save_dir)

    def _init_monitors(self, given_monitors):
        if given_monitors is None:
            return []
        else:
            return given_monitors

    def _setup_logging(self, logging_level):
        logging.basicConfig(filename=self._save_path / "experiment.log",
                            format="%(levelname)s: %(message)s",
                            level=logging_level)

    def _setup_save_path(self, save_dir):
        save_path = Path(save_dir)
        try:
            save_path.mkdir(exist_ok=False)
        except FileExistsError:
            raise ExperimentError(f"Save dir '{save_dir}' already exists.")
        return save_path

    def run(self):
        self._lcs.train(self._monitors)
        self._save()

    def _save(self):
        for monitor in self._monitors:
            monitor.save(self._save_path)
        self._save_run_script()
        self._save_parametrization()
        self._save_lib_version_info()
        self._save_final_population()

    def _save_run_script(self):
        run_script_path = Path(__main__.__file__)
        shutil.copyfile(run_script_path, self._save_path / "run_script.py")

    def _save_parametrization(self):
        with open(self._save_path / "params.txt", "w") as fp:
            for parametrized in (self._env, self._alg, self):
                fp.write(parametrized.get_parametrization_as_str())
                fp.write("\n")

    def _save_lib_version_info(self):
        pass

    def _save_final_population(self):
        pass

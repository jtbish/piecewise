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
                 use_population_monitor=False,
                 population_monitor_freq=1,
                 use_loop_monitor=False,
                 logging_level=logging.INFO):
        self._lcs = LCS(env, alg, num_training_samples, use_population_monitor,
                        population_monitor_freq, use_loop_monitor)
        self._save_path = self._setup_save_path(save_dir)
        self._setup_logging(logging_level, self._save_path)

    def _setup_save_path(self, save_dir):
        save_path = Path(save_dir)
        try:
            save_path.mkdir(exist_ok=False)
        except FileExistsError:
            raise ExperimentError(f"Save dir '{save_dir}' already exists.")
        return save_path

    def _setup_logging(self, logging_level, save_path):
        logging.basicConfig(filename=save_path / "experiment.log",
                            format="%(levelname)s: %(message)s",
                            level=logging_level)

    def run(self):
        trained_population = self._lcs.train()
        return trained_population

    def save(self):
        self._lcs.save_monitor_data(self._save_path)
        self._lcs.save_trained_population(self._save_path)
        self._save_run_script()
        self._save_param_config()
        self._save_lib_version_info()

    def _save_run_script(self):
        run_script_path = Path(__main__.__file__)
        shutil.copyfile(run_script_path, self._save_path / "run_script.py")

    def _save_param_config(self):
        pass

    def _save_lib_version_info(self):
        pass

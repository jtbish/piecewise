import logging
import pickle
import shutil
import subprocess
from pathlib import Path

import __main__
from piecewise.error.experiment_error import ExperimentError

from .trainer import Trainer


class Experiment:
    def __init__(self,
                 name,
                 env,
                 lcs,
                 num_training_samples,
                 use_lcs_monitor=False,
                 lcs_monitor_freq=1,
                 use_loop_monitor=False,
                 logging_level=logging.INFO,
                 var_args=None):
        self._trainer = Trainer(env, lcs, num_training_samples,
                                use_lcs_monitor, lcs_monitor_freq,
                                use_loop_monitor)
        self._save_path = self._setup_save_path(name)
        self._setup_logging(logging_level, self._save_path)
        self._var_args = var_args

        self._trained_lcs = None

    def _setup_save_path(self, name):
        save_path = Path(name)
        try:
            save_path.mkdir(exist_ok=False)
        except FileExistsError:
            raise ExperimentError(f"Save path '{save_path}' already exists.")
        return save_path

    def _setup_logging(self, logging_level, save_path):
        logging.basicConfig(filename=save_path / "experiment.log",
                            format="%(levelname)s: %(message)s",
                            level=logging_level)

    def run(self):
        self._trained_lcs = self._trainer.train_lcs()

    def save_results(self):
        self._trainer.save_monitor_data(self._save_path)
        self._save_trained_lcs()
        self._save_run_script()
        self._save_var_args()
        self._save_lib_version_info()
        self._save_python_env_info()

    def _save_trained_lcs(self):
        with open(self._save_path / "trained_lcs.pkl", "wb") as fp:
            pickle.dump(self._trained_lcs, fp)

    def _save_run_script(self):
        run_script_path = Path(__main__.__file__)
        shutil.copyfile(run_script_path, self._save_path / "run_script.py")

    def _save_var_args(self):
        if self._var_args is not None:
            with open(self._save_path / "var_args.txt", "w") as fp:
                fp.write(str(self._var_args))

    def _save_lib_version_info(self):
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                stdout=subprocess.PIPE)
        return_val = result.stdout.decode("utf-8")
        exit_status = result.returncode
        with open(self._save_path / "lib_version_info.txt", "w") as fp:
            fp.write(f"This is Piecewise, HEAD @ {return_val}")

    def _save_python_env_info(self):
        # TODO
        pass

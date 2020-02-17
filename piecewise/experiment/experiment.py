import functools
import logging
import pickle
import shutil
from pathlib import Path

import __main__
from piecewise.constants import EPOCH_NUM_MIN, TIME_STEP_MIN
from piecewise.error.experiment_error import ExperimentError
from piecewise.monitor import NullMonitor


def try_update_monitor(method):
    @functools.wraps(method)
    def _update_monitor(self):
        result = method(self)

        experiment = self
        experiment._monitor.try_update(experiment)

        return result

    return _update_monitor


class Experiment:
    def __init__(self,
                 save_dir,
                 env,
                 alg,
                 num_training_samples,
                 monitor=None,
                 logging_level=logging.INFO):
        self._save_path = self._setup_save_dir(save_dir)
        self._env = env
        self._alg = alg
        self._num_training_samples = num_training_samples
        self._monitor = self._init_monitor(monitor)
        self._setup_logging(logging_level)
        self._time_step = TIME_STEP_MIN
        self._epoch_num = EPOCH_NUM_MIN
        self._population = None
        self._finished_training = None
        self._latest_return = None

    def _init_monitor(self, monitor):
        if monitor is None:
            monitor = NullMonitor()
        return monitor

    def _setup_logging(self, logging_level):
        logging.basicConfig(filename=self._save_path / "experiment.log",
                            format="%(levelname)s: %(message)s",
                            level=logging_level)

    @property
    def population(self):
        return self._population

    @property
    def time_step(self):
        return self._time_step

    @property
    def latest_return(self):
        return self._latest_return

    def run(self):
        self._perform_training()

    def _perform_training(self):
        logging.info("Starting training")
        self._finished_training = False
        while not self._finished_training:
            self._train_single_epoch()
            self._epoch_num += 1
        logging.info("Finished training")

    @try_update_monitor
    def _train_single_epoch(self):
        logging.info(f"Epoch {self._epoch_num}")
        self._latest_return = 0
        self._env.reset()
        while not self._env.is_terminal() and not self._finished_training:
            self._train_single_time_step()
            self._time_step += 1
            self._finished_training = self._is_finished_training()

    def _train_single_time_step(self):
        logging.info(f"\nTime step {self._time_step}")
        situation = self._get_situation()
        logging.info(f"Situation: {situation}")
        (action, did_explore) = self._alg.train_query(situation,
                                                      self._time_step)
        logging.info(f"Action: {action}")
        env_response = self._env.act(action)
        logging.info(f"Response: {env_response}")
        self._latest_return += env_response.reward
        self._population = self._alg.train_update(env_response)

    def _get_situation(self):
        obs = self._env.observe()
        return obs

    def _is_finished_training(self):
        return self._time_step == self._num_training_samples

    def save(self):
        monitor_results = self._monitor.query()
        self._save_monitor_results_as_pickle_file(monitor_results)
        self._save_population_as_pickle_file()
        self._save_run_script()

    def _setup_save_dir(self, save_dir):
        save_path = Path(save_dir)
        try:
            save_path.mkdir(exist_ok=False)
        except FileExistsError:
            raise ExperimentError(f"Save dir '{save_dir}' already exists.")
        return save_path

    def _save_monitor_results_as_pickle_file(self, monitor_results):
        with open(self._save_path / "monitor_results.pkl", "wb") as fp:
            pickle.dump(monitor_results, fp)

    def _save_population_as_pickle_file(self):
        with open(self._save_path / "population.pkl", "wb") as fp:
            pickle.dump(self._population, fp)

    def _save_run_script(self):
        run_script_path = Path(__main__.__file__)
        shutil.copyfile(run_script_path, self._save_path / "run_script.py")

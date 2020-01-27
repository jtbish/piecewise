import logging
import pickle
import shutil
from pathlib import Path

import __main__
from piecewise.error.experiment_error import ExperimentError
from piecewise.monitor import Monitor

TIME_STEP_MIN = 0


class Experiment:
    def __init__(self,
                 save_dir,
                 env,
                 alg,
                 num_training_epochs,
                 monitor_items=None,
                 logging_level=logging.INFO):
        self._save_path = self._setup_save_dir(save_dir)
        self._env = env
        self._alg = alg
        self._num_training_epochs = num_training_epochs
        self._monitor = self._init_monitor(monitor_items)
        self._setup_logging(logging_level)
        self._time_step = TIME_STEP_MIN
        self._population = None

    def _init_monitor(self, monitor_items):
        if monitor_items is None:
            monitor_items = []
        return Monitor(monitor_items)

    def _setup_logging(self, logging_level):
        logging.basicConfig(filename=self._save_path / "experiment.log",
                            level=logging_level)

    @property
    def population(self):
        return self._population

    def run(self):
        for epoch_num in range(self._num_training_epochs):
            logging.info(f"Starting training epoch {epoch_num}")
            self._train_single_epoch()
            self._monitor.update(self)

    def _train_single_epoch(self):
        self._env.reset()
        while not self._env.is_terminal():
            self._train_single_time_step()
            self._time_step += 1

    def _train_single_time_step(self):
        situation = self._get_situation()
        action = self._alg.train_query(situation, self._time_step)
        env_response = self._env.act(action)
        self._population = self._alg.train_update(env_response)

    def _get_situation(self):
        obs = self._env.observe()
        return obs

    def calc_performance(self, strat):
        valid_strats = ("accuracy", "return")
        if strat == "accuracy":
            return self._calc_accuracy()
        elif strat == "return":
            return self._calc_return()
        else:
            raise ExperimentError(
                "Invalid performance calculation strategy: "
                f"{strat}. Valid strategies are: {valid_strats}")

    def _calc_accuracy(self):
        self._env.reset()
        prediction_results = []
        while not self._env.is_terminal():
            situation = self._get_situation()
            action = self._alg.test_query(situation)
            env_response = self._env.act(action)
            prediction_results.append(env_response.was_correct_action)
        accuracy = \
            (prediction_results.count(True) / len(prediction_results)) * 100
        return accuracy

    def _calc_return(self):
        self._env.reset()
        rewards = []
        while not self._env.is_terminal():
            situation = self._get_situation()
            action = self._alg.test_query(situation)
            env_response = self._env.act(action)
            rewards.append(env_response.reward)
        return sum(rewards)

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

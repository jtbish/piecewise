import logging
import pickle
from collections import namedtuple

from piecewise.constants import EPOCH_NUM_MIN, TIME_STEP_MIN
from piecewise.monitor import Monitor, NullMonitor

LoopData = namedtuple("LoopData",
                      ["situation", "alg_response", "env_response"])


class LCS:
    def __init__(self, env, alg, num_training_samples, use_population_monitor,
                 population_monitor_freq, use_loop_monitor):
        self._env = env
        self._alg = alg
        self._num_training_samples = num_training_samples
        self._population_monitor = \
            self._init_population_monitor(use_population_monitor,
                                          population_monitor_freq)
        self._loop_monitor = self._init_loop_monitor(use_loop_monitor)

        self._time_step = TIME_STEP_MIN
        self._epoch_num = EPOCH_NUM_MIN

        self._population = None
        self._loop_data = None

    def _init_population_monitor(self, use_population_monitor,
                                 population_monitor_freq):
        if use_population_monitor:
            return Monitor("population", update_freq=population_monitor_freq)
        else:
            return NullMonitor()

    def _init_loop_monitor(self, use_loop_monitor):
        if use_loop_monitor:
            return Monitor("loop_data", update_freq=1)
        else:
            return NullMonitor()

    def get_parametrization(self):
        pass

    def train(self):
        logging.info("Starting training")
        while not self._is_finished_training():
            self._train_single_epoch()
            self._epoch_num += 1
        logging.info("Finished training")
        return self._population

    def _train_single_epoch(self):
        logging.info(f"Epoch {self._epoch_num}")
        self._env.reset()
        while not self._env.is_terminal() and not self._is_finished_training():
            self._train_single_time_step()
            self._time_step += 1
            self._update_monitors()

    def _train_single_time_step(self):
        logging.info(f"Time step {self._time_step}")

        situation = self._get_situation()
        logging.info(f"Situation: {situation}")

        alg_response = self._alg.train_query(situation, self._time_step)
        logging.info(f"Alg response: {alg_response}")

        action = alg_response.action
        env_response = self._env.act(action)
        logging.info(f"Env response: {env_response}")

        self._population = self._alg.train_update(env_response)
        self._loop_data = LoopData(situation=situation,
                                   alg_response=alg_response,
                                   env_response=env_response)

    def _get_situation(self):
        obs = self._env.observe()
        return obs

    def _update_monitors(self):
        self._population_monitor.update(self._time_step, self._population)
        self._loop_monitor.update(self._time_step, self._loop_data)

    def _is_finished_training(self):
        return self._time_step == self._num_training_samples

    def save_monitor_data(self, save_path):
        for monitor in (self._population_monitor, self._loop_monitor):
            monitor.save(save_path)

    def save_trained_population(self, save_path):
        with open(save_path / "trained_population.pkl", "wb") as fp:
            pickle.dump(self._population, fp)

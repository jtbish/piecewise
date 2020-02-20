import logging
from collections import namedtuple

from piecewise.constants import EPOCH_NUM_MIN, TIME_STEP_MIN
from piecewise.monitor import Monitor, NullMonitor

LoopData = namedtuple("LoopData",
                      ["situation", "lcs_response", "env_response"])


class Trainer:
    def __init__(self, env, lcs, num_training_samples, use_lcs_monitor,
                 lcs_monitor_freq, use_loop_monitor):
        self._env = env
        self._lcs = lcs
        self._num_training_samples = num_training_samples
        self._lcs_monitor = \
            self._init_lcs_monitor(use_lcs_monitor,
                                   lcs_monitor_freq)
        self._loop_monitor = self._init_loop_monitor(use_loop_monitor)

        self._time_step = TIME_STEP_MIN
        self._epoch_num = EPOCH_NUM_MIN

        self._loop_data = None

    def _init_lcs_monitor(self, use_lcs_monitor, lcs_monitor_freq):
        if use_lcs_monitor:
            return Monitor("lcs", update_freq=lcs_monitor_freq)
        else:
            return NullMonitor()

    def _init_loop_monitor(self, use_loop_monitor):
        if use_loop_monitor:
            return Monitor("loop_data", update_freq=1)
        else:
            return NullMonitor()

    def train_lcs(self):
        logging.info("Starting training")
        while not self._is_finished_training():
            self._train_single_epoch()
            self._epoch_num += 1
        logging.info("Finished training")
        return self._lcs

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

        lcs_response = self._lcs.train_query(situation, self._time_step)
        logging.info(f"LCS response: {lcs_response}")

        action = lcs_response.action
        env_response = self._env.act(action)
        logging.info(f"Env response: {env_response}")

        self._loop_data = LoopData(situation=situation,
                                   lcs_response=lcs_response,
                                   env_response=env_response)

    def _get_situation(self):
        return self._env.observe()

    def _update_monitors(self):
        self._lcs_monitor.update(self._time_step, self._lcs)
        self._loop_monitor.update(self._time_step, self._loop_data)

    def _is_finished_training(self):
        return self._time_step == self._num_training_samples

    def save_monitor_data(self, save_path):
        for monitor in (self._lcs_monitor, self._loop_monitor):
            monitor.save(save_path)

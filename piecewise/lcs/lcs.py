import functools
import logging
from collections import namedtuple

from piecewise.constants import EPOCH_NUM_MIN, TIME_STEP_MIN
from piecewise.util import ParametrizedMixin

LoopData = namedtuple("LoopData",
                      ["situation", "alg_response", "env_response"])


def update_monitors_after(method):
    @functools.wraps(method)
    def _update_monitors_after(self):
        result = method(self)

        lcs = self
        for monitor in lcs._monitors:
            monitor.update(lcs)

        return result

    return _update_monitors_after


class LCS(ParametrizedMixin):
    def __init__(self, env, alg, num_training_samples):
        self._env = env
        self._alg = alg
        self._num_training_samples = num_training_samples

        self._time_step = TIME_STEP_MIN
        self._epoch_num = EPOCH_NUM_MIN
        self._population = None
        self._finished_training = False
        self._loop_data = None

        self.is_parametrized_by(num_training_samples=num_training_samples)

    # TODO why do these properties even exist? Bad OOP?

    @property
    def time_step(self):
        return self._time_step

    @property
    def population(self):
        return self._population

    @property
    def loop_data(self):
        return self._loop_data

    def train(self, monitors):
        logging.info("Starting training")
        while not self._finished_training:
            self._train_single_epoch()
            self._epoch_num += 1
        logging.info("Finished training")

    def _train_single_epoch(self):
        logging.info(f"Epoch {self._epoch_num}")
        self._env.reset()
        while not self._env.is_terminal() and not self._finished_training:
            self._train_single_time_step()
            self._time_step += 1
            self._finished_training = self._is_finished_training()

    @update_monitors_after
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

        # TODO return this instead?
        self._loop_data = LoopData(situation=situation,
                                   alg_response=alg_response,
                                   env_response=env_response)

    def _get_situation(self):
        obs = self._env.observe()
        return obs

    def _is_finished_training(self):
        return self._time_step == self._num_training_samples

    def _get_parametrization_as_str(self):
        pass

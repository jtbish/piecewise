from piecewise.encoding import null_encoding
from piecewise.monitor import Monitor

TIME_STEP_MIN = 0


class Experiment:
    def __init__(self,
                 env,
                 alg,
                 num_training_epochs,
                 encoding=None,
                 monitor_items=None,
                 logging="verbose"):
        self._env = env
        self._alg = alg
        self._num_training_epochs = num_training_epochs
        self._encoding = self._init_encoding(encoding)
        self._monitor = self._init_monitor(monitor_items)
        self._time_step = TIME_STEP_MIN
        self._population = None

    def _init_encoding(self, encoding):
        if encoding is None:
            encoding = null_encoding
        return encoding

    def _init_monitor(self, monitor_items):
        if monitor_items is None:
            monitor_items = []
        return Monitor(monitor_items)

    @property
    def population(self):
        return self._population

    def run(self):
        for epoch_num in range(self._num_training_epochs):
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
        return self._encoding(obs)

    def calc_performance(self, strat):
        if strat == "accuracy":
            return self._calc_accuracy()
        elif strat == "return":
            return self._calc_return()
        else:
            # TODO change
            raise Exception

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

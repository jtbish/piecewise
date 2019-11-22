from piecewise.monitor import NullMonitor


class LCS:
    def __init__(self, env, codec, rule_repr, algorithm, monitor=None):
        self._env = env
        self._codec = codec
        self._rule_repr = rule_repr
        self._algorithm = algorithm
        if monitor is None:
            self._monitor = NullMonitor()
        else:
            self._monitor = monitor

        self._epoch_num = 0
        self._time_step = 0
        self._population = None

    def train(self, num_epochs=1):
        while self._epoch_num < num_epochs:
            self._population = self._exec_train_epoch()
            self._monitor.update(self)
            self._monitor.report()
            self._epoch_num += 1
        return self._population

    def _exec_train_epoch(self):
        self._env.reset()
        while not self._env.is_terminal():
            population = self._exec_train_time_step()
            self._time_step += 1
        return population

    def _exec_train_time_step(self):
        situation = self._get_situation()
        action = self._algorithm.train_query(situation, self._time_step)
        env_response = self._env.act(action)
        population = self._algorithm.train_update(env_response)
        return population

    # TODO generalise to RL
    def calc_training_performance(self):
        self._env.reset()
        results = []
        while not self._env.is_terminal():
            situation = self._get_situation()
            action = self._algorithm.test_query(situation)
            env_response = self._env.act(action)
            results.append(env_response.was_correct_action)
        training_accuracy = (results.count(True) / len(results)) * 100
        return training_accuracy

    def _get_situation(self):
        obs = self._env.observe()
        return self._codec.encode(obs)

    @property
    def epoch_num(self):
        return self._epoch_num

    @property
    def time_step(self):
        return self._time_step

    @property
    def population(self):
        return self._population

    @property
    def rule_repr(self):
        return self._rule_repr

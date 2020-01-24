TIME_STEP_MIN = 0


class Experiment:
    def __init__(self, env, alg, num_training_epochs, logging="verbose"):
        self._env = env
        self._alg = alg
        self._num_training_epochs = num_training_epochs
        self._time_step = TIME_STEP_MIN
        self._population = None

    def run(self):
        for epoch_num in range(self._num_training_epochs):
            self._train_single_epoch()
            self._exec_monitor_callbacks()

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
        return self._env.observe()

    def _exec_monitor_callbacks(self):
        pass

    def archive(self):
        raise NotImplementedError

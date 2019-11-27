import abc

TIME_STEP_MIN = 1


class LCS(metaclass=abc.ABCMeta):
    def __init__(self, env, codec, algorithm):
        self._env = env
        self._codec = codec
        self._algorithm = algorithm

        self._time_step = TIME_STEP_MIN
        self._population = None

    @abc.abstractmethod
    def calc_training_performance(self):
        raise NotImplementedError

    @property
    def population(self):
        return self._population

    def train_single_epoch(self):
        self._env.reset()
        while not self._env.is_terminal():
            self._population = self._train_single_time_step()
            self._time_step += 1
        return self._population

    def _train_single_time_step(self):
        situation = self._get_situation()
        action = self._algorithm.train_query(situation, self._time_step)
        env_response = self._env.act(action)
        population = self._algorithm.train_update(env_response)
        return population

    def _get_situation(self):
        obs = self._env.observe()
        return self._codec.encode(obs)

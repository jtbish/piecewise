from piecewise.util.classifier_set_stats import summarise_population


class LCS:
    def __init__(self, env, codec, rule_repr, algorithm):
        self._env = env
        self._codec = codec
        self._rule_repr = rule_repr
        self._algorithm = algorithm
        self._time_step = 0

    def train(self, num_epochs=1, monitor=False):
        for epoch_num in range(num_epochs):
            population = self._exec_train_epoch(epoch_num)
            if monitor:
                self._calc_training_performance(epoch_num)
                summary = summarise_population(population, self._rule_repr,
                                               self._time_step)
                self._print_population_summary(summary)
                self._print_population_state(population)
                print("\n")
        return population

    def _exec_train_epoch(self, epoch_num):
        self._env.reset()
        while not self._env.is_terminal():
            population = self._exec_train_time_step()
            self._time_step += 1
        return population

    def _exec_train_time_step(self):
        situation = self._get_situation()
        action = self._algorithm.train_query(situation, self._time_step)
        env_response = self._env.act(action)
        env_is_terminal = self._env.is_terminal()
        population = self._algorithm.train_update(env_response,
                                                  env_is_terminal)
        return population

    def _get_situation(self):
        obs = self._env.observe()
        return self._codec.encode(obs)

    # change below here
    def _calc_training_performance(self, epoch_num):
        self._env.reset()
        results = []
        while not self._env.is_terminal():
            situation = self._get_situation()
            action = self._algorithm.test_query(situation)
            env_response = self._env.act(action)
            results.append(env_response.was_correct_action)
        training_accuracy = (results.count(True) / len(results)) * 100
        print(f"Training performance at epoch {epoch_num}: "
              f"{training_accuracy:.2f}%")

    def _print_population_summary(self, summary):
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    def _print_population_state(self, population):
        print("Population tracking:")
        print(dict(population._state.query()))

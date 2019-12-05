from ..lcs import LCS


class ClassificationLCS(LCS):
    def calc_training_performance(self):
        """Calculates accuracy on training set."""
        self._env.reset()
        correctness_results = []
        while not self._env.is_terminal():
            situation = self._get_situation()
            action = self._alg.test_query(situation)
            env_response = self._env.act(action)
            correctness_results.append(env_response.was_correct_action)
        training_accuracy = (correctness_results.count(True) /
                             len(correctness_results)) * 100
        return training_accuracy

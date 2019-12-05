from ..lcs import LCS


class ReinforcementLCS(LCS):
    def calc_training_performance(self):
        """Calculates undisconunted return over one episode."""
        self._env.reset()
        return_achieved = 0
        while not self._env.is_terminal():
            situation = self._get_situation()
            action = self._alg.test_query(situation)
            env_response = self._env.act(action)
            return_achieved += env_response.reward
        return return_achieved

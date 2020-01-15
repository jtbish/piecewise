import numpy as np


class MonitorOutput:
    def __init__(self):
        self._trait_history = {}

    def record(self, trait_name, trial_data):
        self._trait_history[trait_name] = np.array(trial_history)

    def extend(self, trait_name, more_trial_history):
        existing_trait_history = self[trait_name]
        extended_trait_history = np.vstack((existing_trials, more_trial_data))
        self._trait_history[trait_name] = extended_trial_history

    def __getitem__(self, key):
        trait_name = key
        return self._dict[trait_name]

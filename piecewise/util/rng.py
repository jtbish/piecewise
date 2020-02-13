import numpy as np


def init_np_random_state(seed):
    seed = int(seed)
    return np.random.RandomState(seed)

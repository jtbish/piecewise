from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng


def _swap_vec_elems(first_vec, second_vec, swap_idx):
    first_vec[swap_idx], second_vec[swap_idx] = \
            second_vec[swap_idx], first_vec[swap_idx]


class TwoPointCrossover:
    def __call__(self, first_vec, second_vec):
        """Based on APPLY CROSSOVER function from 'An Algorithmic Description
        of XCS' (Butz and Wilson, 2002)."""
        assert len(first_vec) == len(second_vec)
        (first_idx, second_idx) = \
            self._choose_random_crossover_idxs(len(first_vec))
        (first_idx,
         second_idx) = self._order_crossover_idxs(first_idx, second_idx)
        assert first_idx <= second_idx
        self._crossover_vecs(first_vec, second_vec, first_idx, second_idx)

    def _choose_random_crossover_idxs(self, vec_len):
        num_idxs = 2
        return tuple([get_rng().randint(0, vec_len) for _ in range(num_idxs)])

    def _order_crossover_idxs(self, first_idx, second_idx):
        return (min(first_idx, second_idx), max(first_idx, second_idx))

    def _crossover_vecs(self, first_vec, second_vec, first_idx, second_idx):
        for swap_idx in range(first_idx, second_idx):
            _swap_vec_elems(first_vec, second_vec, swap_idx)


class UniformCrossover:
    def __call__(self, first_vec, second_vec):
        assert len(first_vec) == len(second_vec)
        for swap_idx in range(0, len(first_vec)):
            should_swap = get_rng().rand() < get_hyperparam("upsilon")
            if should_swap:
                _swap_vec_elems(first_vec, second_vec, swap_idx)

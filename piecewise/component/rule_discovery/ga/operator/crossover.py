import random


class TwoPointCrossover:
    def __call__(self, child_one, child_two):
        """Based on APPLY CROSSOVER function from 'An Algorithmic Description
        of XCS' (Butz and Wilson, 2002)."""
        assert len(child_one.condition) == len(child_two.condition)
        (first_idx, second_idx) = \
            self._choose_random_crossover_idxs(len(child_one.condition))
        (first_idx,
         second_idx) = self._order_crossover_idxs(first_idx, second_idx)
        self._crossover_child_conditions(child_one, child_two, first_idx,
                                         second_idx)

    def _choose_random_crossover_idxs(self, condition_len):
        num_idxs = 2
        return tuple(
            [random.randint(0, condition_len - 1) for _ in range(num_idxs)])

    def _order_crossover_idxs(self, first_idx, second_idx):
        return (min(first_idx, second_idx), max(first_idx, second_idx))

    def _crossover_child_conditions(self, child_one, child_two, first_idx,
                                    second_idx):
        for swap_idx in range(first_idx, second_idx):
            self._swap_condition_elements(child_one, child_two, swap_idx)

    def _swap_condition_elements(self, child_one, child_two, swap_idx):
        child_one.condition[swap_idx], child_two.condition[swap_idx] = \
                child_two.condition[swap_idx], child_one.condition[swap_idx]

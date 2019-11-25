import random

import pytest

from piecewise.dtype import Population
from piecewise.error.classifier_set_error import MemberNotFoundError
from piecewise.error.population_error import InvalidSizeError


@pytest.fixture
def mock_deletion_strat(mocker):
    return mocker.MagicMock()


@pytest.fixture
def random_deletion_strat(mocker):
    deletion_strat = mocker.MagicMock()
    deletion_strat.side_effect = lambda population: \
        random.choice(population._members)
    return deletion_strat


class TestPopulation:
    def test_bad_init_zero_max_micros(self, mock_deletion_strat):
        with pytest.raises(InvalidSizeError):
            Population(max_micros=0, deletion_strat=mock_deletion_strat)

    def test_bad_init_neg_max_micros(self, mock_deletion_strat):
        with pytest.raises(InvalidSizeError):
            Population(max_micros=-1, deletion_strat=mock_deletion_strat)

    def test_add(self, mock_deletion_strat, make_mock_microclassifier):
        population = Population(max_micros=1,
                                deletion_strat=mock_deletion_strat)
        mock_microclassifier = make_mock_microclassifier()
        population.add(mock_microclassifier)
        assert population.num_micros() == 1
        assert population.num_macros() == 1

    def test_insert_no_absorb(self, mock_deletion_strat,
                              make_mock_microclassifier):
        population = Population(max_micros=2,
                                deletion_strat=mock_deletion_strat)
        mock_microclassifier = make_mock_microclassifier()
        diff_mock_microclassifier = make_mock_microclassifier()
        population.insert(mock_microclassifier)
        population.insert(diff_mock_microclassifier)
        assert population.num_micros() == 2
        assert population.num_macros() == 2

    def test_insert_force_absorb_microclassifier(self, mock_deletion_strat,
                                                 mock_microclassifier):
        population = Population(max_micros=2,
                                deletion_strat=mock_deletion_strat)
        population.insert(mock_microclassifier)
        population.insert(mock_microclassifier)
        assert population.num_micros() == 2
        assert population.num_macros() == 1

    def test_insert_force_absorb_macroclassifier(self, mock_deletion_strat,
                                                 mock_microclassifier,
                                                 make_mock_macroclassifier,
                                                 mocker):
        population = Population(max_micros=3,
                                deletion_strat=mock_deletion_strat)
        mock_macroclassifier = make_mock_macroclassifier(numerosity=2)

        shared_rule = mocker.MagicMock()
        mock_microclassifier.rule = shared_rule
        mock_macroclassifier.rule = shared_rule

        population.insert(mock_microclassifier)
        population.insert(mock_macroclassifier)
        assert population.num_micros() == 3
        assert population.num_macros() == 1

    def test_insert_trigger_single_deletion(self, random_deletion_strat,
                                            mock_microclassifier,
                                            make_mock_macroclassifier):
        population = Population(max_micros=2,
                                deletion_strat=random_deletion_strat)
        mock_macroclassifier = make_mock_macroclassifier(numerosity=2)
        population.insert(mock_microclassifier)
        population.insert(mock_macroclassifier)
        assert population.num_micros() == 2
        random_deletion_strat.assert_called_once()

    def test_insert_trigger_double_deletion(self, random_deletion_strat,
                                            mock_microclassifier,
                                            make_mock_macroclassifier):
        population = Population(max_micros=2,
                                deletion_strat=random_deletion_strat)
        mock_macroclassifier = make_mock_macroclassifier(numerosity=3)
        population.insert(mock_microclassifier)
        population.insert(mock_macroclassifier)
        assert population.num_micros() == 2
        expected_num_deletions = 2
        assert random_deletion_strat.call_count == expected_num_deletions

    def test_duplicate_single_copy(self, mock_deletion_strat,
                                   mock_microclassifier):
        population = Population(max_micros=2,
                                deletion_strat=mock_deletion_strat)
        population.insert(mock_microclassifier)
        population.duplicate(mock_microclassifier, num_copies=1)
        assert population.num_micros() == 2
        assert population.num_macros() == 1

    def test_duplicate_two_copies(self, mock_deletion_strat,
                                  mock_microclassifier):
        population = Population(max_micros=3,
                                deletion_strat=mock_deletion_strat)
        population.insert(mock_microclassifier)
        population.duplicate(mock_microclassifier, num_copies=2)
        assert population.num_micros() == 3
        assert population.num_macros() == 1

    def test_duplicate_non_member(self, mock_deletion_strat,
                                  make_mock_microclassifier):
        population = Population(max_micros=2,
                                deletion_strat=mock_deletion_strat)
        mock_microclassifier = make_mock_microclassifier()
        diff_mock_microclassifier = make_mock_microclassifier()
        population.insert(mock_microclassifier)
        with pytest.raises(MemberNotFoundError):
            population.duplicate(diff_mock_microclassifier, num_copies=1)
        assert population.num_micros() == 1
        assert population.num_macros() == 1

    def test_replace_fail_replacer_non_member(self, mock_deletion_strat,
                                              make_mock_microclassifier):
        population = Population(max_micros=1,
                                deletion_strat=mock_deletion_strat)
        replacee = make_mock_microclassifier()
        population.insert(replacee)
        replacer = make_mock_microclassifier()
        with pytest.raises(MemberNotFoundError):
            population.replace(replacee, replacer)

    def test_replace_fail_replacee_non_member(self, mock_deletion_strat,
                                              make_mock_microclassifier):
        population = Population(max_micros=1,
                                deletion_strat=mock_deletion_strat)
        replacee = make_mock_microclassifier()
        replacer = make_mock_microclassifier()
        population.insert(replacer)
        with pytest.raises(MemberNotFoundError):
            population.replace(replacee, replacer)

    def test_replace_succeed_both_micros(self, mock_deletion_strat,
                                         make_mock_microclassifier):
        population = Population(max_micros=2,
                                deletion_strat=mock_deletion_strat)
        replacee = make_mock_microclassifier()
        replacer = make_mock_microclassifier()
        population.insert(replacee)
        population.insert(replacer)
        assert population.num_micros() == 2
        assert population.num_macros() == 2
        population.replace(replacee, replacer)
        assert population.num_micros() == 2
        assert population.num_macros() == 1

    def test_replace_succeed_replacee_is_macro(self, mock_deletion_strat,
                                               make_mock_microclassifier,
                                               make_mock_macroclassifier):
        population = Population(max_micros=3,
                                deletion_strat=mock_deletion_strat)
        replacee = make_mock_macroclassifier(numerosity=2)
        replacer = make_mock_microclassifier()
        population.insert(replacee)
        population.insert(replacer)
        assert population.num_micros() == 3
        assert population.num_macros() == 2
        population.replace(replacee, replacer)
        assert population.num_micros() == 3
        assert population.num_macros() == 1

    def test_replace_succeed_replacer_is_macro(self, mock_deletion_strat,
                                               make_mock_microclassifier,
                                               make_mock_macroclassifier):
        population = Population(max_micros=3,
                                deletion_strat=mock_deletion_strat)
        replacee = make_mock_microclassifier()
        replacer = make_mock_macroclassifier(numerosity=2)
        population.insert(replacee)
        population.insert(replacer)
        assert population.num_micros() == 3
        assert population.num_macros() == 2
        population.replace(replacee, replacer)
        assert population.num_micros() == 3
        assert population.num_macros() == 1

    def test_replace_succeed_replacer_both_micro(self, mock_deletion_strat,
                                                 make_mock_macroclassifier):
        population = Population(max_micros=4,
                                deletion_strat=mock_deletion_strat)
        replacee = make_mock_macroclassifier(numerosity=2)
        replacer = make_mock_macroclassifier(numerosity=2)
        population.insert(replacee)
        population.insert(replacer)
        assert population.num_micros() == 4
        assert population.num_macros() == 2
        population.replace(replacee, replacer)
        assert population.num_micros() == 4
        assert population.num_macros() == 1

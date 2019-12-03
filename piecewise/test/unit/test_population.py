import pytest

from piecewise.dtype import Population
from piecewise.error.classifier_set_error import MemberNotFoundError
from piecewise.error.population_error import InvalidSizeError


class TestPopulation:
    def test_bad_init_zero_max_micros(self):
        with pytest.raises(InvalidSizeError):
            Population(max_micros=0)

    def test_bad_init_neg_max_micros(self):
        with pytest.raises(InvalidSizeError):
            Population(max_micros=-1)

    def test_add(self, make_mock_microclassifier):
        population = Population(max_micros=1)
        mock_microclassifier = make_mock_microclassifier()
        population.add(mock_microclassifier)
        assert population.num_micros == 1
        assert population.num_macros == 1

    def test_insert_no_absorb(self, make_mock_microclassifier):
        population = Population(max_micros=2)
        mock_microclassifier = make_mock_microclassifier()
        diff_mock_microclassifier = make_mock_microclassifier()
        population.insert(mock_microclassifier)
        population.insert(diff_mock_microclassifier)
        assert population.num_micros == 2
        assert population.num_macros == 2

    def test_insert_force_absorb_microclassifier(self, mock_microclassifier):
        population = Population(max_micros=2)
        population.insert(mock_microclassifier)
        population.insert(mock_microclassifier)
        assert population.num_micros == 2
        assert population.num_macros == 1

    def test_insert_force_absorb_macroclassifier(self, mock_microclassifier,
                                                 make_mock_macroclassifier,
                                                 mocker):
        population = Population(max_micros=3)
        mock_macroclassifier = make_mock_macroclassifier(numerosity=2)

        shared_rule = mocker.MagicMock()
        mock_microclassifier.rule = shared_rule
        mock_macroclassifier.rule = shared_rule

        population.insert(mock_microclassifier)
        population.insert(mock_macroclassifier)
        assert population.num_micros == 3
        assert population.num_macros == 1

    def test_duplicate_single_copy(self, mock_microclassifier):
        population = Population(max_micros=2)
        population.insert(mock_microclassifier)
        population.duplicate(mock_microclassifier, num_copies=1)
        assert population.num_micros == 2
        assert population.num_macros == 1

    def test_duplicate_two_copies(self, mock_microclassifier):
        population = Population(max_micros=3)
        population.insert(mock_microclassifier)
        population.duplicate(mock_microclassifier, num_copies=2)
        assert population.num_micros == 3
        assert population.num_macros == 1

    def test_duplicate_fail_non_member(self, make_mock_microclassifier):
        population = Population(max_micros=2)
        mock_microclassifier = make_mock_microclassifier()
        diff_mock_microclassifier = make_mock_microclassifier()
        population.insert(mock_microclassifier)
        with pytest.raises(MemberNotFoundError):
            population.duplicate(diff_mock_microclassifier, num_copies=1)
        assert population.num_micros == 1
        assert population.num_macros == 1

    def test_replace_fail_replacer_non_member(self, make_mock_microclassifier):
        population = Population(max_micros=1)
        replacee = make_mock_microclassifier()
        population.insert(replacee)
        replacer = make_mock_microclassifier()
        with pytest.raises(MemberNotFoundError):
            population.replace(replacee, replacer)

    def test_replace_fail_replacee_non_member(self, make_mock_microclassifier):
        population = Population(max_micros=1)
        replacee = make_mock_microclassifier()
        replacer = make_mock_microclassifier()
        population.insert(replacer)
        with pytest.raises(MemberNotFoundError):
            population.replace(replacee, replacer)

    def test_replace_succeed_both_micros(self, make_mock_microclassifier):
        population = Population(max_micros=2)
        replacee = make_mock_microclassifier()
        replacer = make_mock_microclassifier()
        population.insert(replacee)
        population.insert(replacer)
        assert population.num_micros == 2
        assert population.num_macros == 2
        population.replace(replacee, replacer)
        assert population.num_micros == 2
        assert population.num_macros == 1

    def test_replace_succeed_replacee_is_macro(self, make_mock_microclassifier,
                                               make_mock_macroclassifier):
        population = Population(max_micros=3)
        replacee = make_mock_macroclassifier(numerosity=2)
        replacer = make_mock_microclassifier()
        population.insert(replacee)
        population.insert(replacer)
        assert population.num_micros == 3
        assert population.num_macros == 2
        population.replace(replacee, replacer)
        assert population.num_micros == 3
        assert population.num_macros == 1

    def test_replace_succeed_replacer_is_macro(self, make_mock_microclassifier,
                                               make_mock_macroclassifier):
        population = Population(max_micros=3)
        replacee = make_mock_microclassifier()
        replacer = make_mock_macroclassifier(numerosity=2)
        population.insert(replacee)
        population.insert(replacer)
        assert population.num_micros == 3
        assert population.num_macros == 2
        population.replace(replacee, replacer)
        assert population.num_micros == 3
        assert population.num_macros == 1

    def test_replace_succeed_replacer_both_micro(self,
                                                 make_mock_macroclassifier):
        population = Population(max_micros=4)
        replacee = make_mock_macroclassifier(numerosity=2)
        replacer = make_mock_macroclassifier(numerosity=2)
        population.insert(replacee)
        population.insert(replacer)
        assert population.num_micros == 4
        assert population.num_macros == 2
        population.replace(replacee, replacer)
        assert population.num_micros == 4
        assert population.num_macros == 1

    def test_delete_fail_non_member(self, make_mock_microclassifier):
        population = Population(max_micros=1)
        mock_microclassifier = make_mock_microclassifier()
        population.insert(mock_microclassifier)
        diff_mock_microclassifier = make_mock_microclassifier()
        with pytest.raises(MemberNotFoundError):
            population.delete(diff_mock_microclassifier)

    def test_delete_succeed_on_micro(self, make_mock_microclassifier):
        population = Population(max_micros=1)
        mock_microclassifier = make_mock_microclassifier()
        population.insert(mock_microclassifier)
        assert population.num_micros == 1
        assert population.num_macros == 1
        population.delete(mock_microclassifier)
        assert population.num_micros == 0
        assert population.num_macros == 0

    def test_delete_succeed_on_macro(self, make_mock_macroclassifier):
        population = Population(max_micros=2)
        mock_macroclassifier = make_mock_macroclassifier(numerosity=2)
        population.insert(mock_macroclassifier)
        assert population.num_micros == 2
        assert population.num_macros == 1
        population.delete(mock_macroclassifier)
        assert population.num_micros == 1
        assert population.num_macros == 1

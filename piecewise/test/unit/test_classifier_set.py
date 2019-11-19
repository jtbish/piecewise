import pytest

from piecewise.dtype import ClassifierSet
from piecewise.error.classifier_set_error import MemberNotFoundError


@pytest.fixture
def classifier_set():
    return ClassifierSet()


class TestAbstractClassifierSetViaClassifierSet:
    def test_num_micros_single_microclassifier(self, classifier_set,
                                               mock_microclassifier):
        classifier_set.add(mock_microclassifier)
        assert classifier_set.num_micros() == 1

    def test_num_micros_multiple_microclassifiers(self, classifier_set,
                                                  make_mock_microclassifier):
        first_microclassifier = make_mock_microclassifier()
        second_microclassifier = make_mock_microclassifier()
        classifier_set.add(first_microclassifier)
        classifier_set.add(second_microclassifier)
        assert classifier_set.num_micros() == 2

    def test_num_micros_single_macroclassifier(self, classifier_set,
                                               make_mock_macroclassifier):
        mock_macroclassifier = make_mock_macroclassifier(numerosity=2)
        classifier_set.add(mock_macroclassifier)
        assert classifier_set.num_micros() == 2

    def test_num_micros_multiple_macroclassifiers(self, classifier_set,
                                                  make_mock_macroclassifier):
        first_macroclassifier = make_mock_macroclassifier(numerosity=2)
        second_macroclassifier = make_mock_macroclassifier(numerosity=2)
        classifier_set.add(first_macroclassifier)
        classifier_set.add(second_macroclassifier)
        assert classifier_set.num_micros() == 4

    def test_num_micros_mixed(self, classifier_set, mock_microclassifier,
                              make_mock_macroclassifier):
        mock_macroclassifier = make_mock_macroclassifier(numerosity=2)
        classifier_set.add(mock_microclassifier)
        classifier_set.add(mock_macroclassifier)
        assert classifier_set.num_micros() == 3

    def test_num_macros_single_microclassifier(self, classifier_set,
                                               mock_microclassifier):
        classifier_set.add(mock_microclassifier)
        assert classifier_set.num_macros() == 1

    def test_num_macros_multiple_microclassifiers(self, classifier_set,
                                                  make_mock_microclassifier):
        first_microclassifier = make_mock_microclassifier()
        second_microclassifier = make_mock_microclassifier()
        classifier_set.add(first_microclassifier)
        classifier_set.add(second_microclassifier)
        assert classifier_set.num_macros() == 2

    def test_num_macros_single_macroclassifier(self, classifier_set,
                                               make_mock_macroclassifier):
        mock_macroclassifier = make_mock_macroclassifier(numerosity=2)
        classifier_set.add(mock_macroclassifier)
        assert classifier_set.num_macros() == 1

    def test_num_macros_multiple_macroclassifiers(self, classifier_set,
                                                  make_mock_macroclassifier):
        first_macroclassifier = make_mock_macroclassifier(numerosity=2)
        second_macroclassifier = make_mock_macroclassifier(numerosity=2)
        classifier_set.add(first_macroclassifier)
        classifier_set.add(second_macroclassifier)
        assert classifier_set.num_macros() == 2

    def test_num_macros_mixed(self, classifier_set, mock_microclassifier,
                              make_mock_macroclassifier):
        mock_macroclassifier = make_mock_macroclassifier(numerosity=2)
        classifier_set.add(mock_microclassifier)
        classifier_set.add(mock_macroclassifier)
        assert classifier_set.num_macros() == 2

    def test_containment_pos_case(self, classifier_set, mock_microclassifier):
        classifier_set.add(mock_microclassifier)
        assert mock_microclassifier in classifier_set

    def test_containment_neg_case(self, classifier_set, mock_microclassifier):
        assert mock_microclassifier not in classifier_set

    def test_iter_empty_case(self):
        empty_classifier_set = ClassifierSet()
        members = [member for member in empty_classifier_set]
        assert members == []

    def test_iter_non_empty_case(self, mock_microclassifier):
        classifier_set = ClassifierSet()
        classifier_set.add(mock_microclassifier)
        members = [member for member in classifier_set]
        assert len(members) == 1
        assert members[0] == mock_microclassifier


class TestClassifierSet:
    def test_add_microclassifier(self, classifier_set, mock_microclassifier):
        classifier_set.add(mock_microclassifier)
        assert classifier_set._members[-1] == mock_microclassifier

    def test_add_macroclassifier(self, classifier_set,
                                 make_mock_macroclassifier):
        mock_macroclassifier = make_mock_macroclassifier(numerosity=2)
        classifier_set.add(mock_macroclassifier)
        assert classifier_set._members[-1] == mock_macroclassifier

    def test_remove_microclassifier(self, classifier_set,
                                    mock_microclassifier):
        classifier_set.add(mock_microclassifier)
        classifier_set.remove(mock_microclassifier)
        assert classifier_set.num_micros() == 0
        assert classifier_set.num_macros() == 0

    def test_remove_macroclassifier(self, classifier_set,
                                    make_mock_macroclassifier):
        mock_macroclassifier = make_mock_macroclassifier(numerosity=2)
        classifier_set.add(mock_macroclassifier)
        classifier_set.remove(mock_macroclassifier)
        assert classifier_set.num_micros() == 0
        assert classifier_set.num_macros() == 0

    def test_remove_non_member(self, classifier_set, mock_microclassifier):
        with pytest.raises(MemberNotFoundError):
            classifier_set.remove(mock_microclassifier)
        assert classifier_set.num_micros() == 0
        assert classifier_set.num_macros() == 0

import pytest

from piecewise.dtype import Classifier
from piecewise.dtype.classifier import (ACTION_SET_SIZE_MIN, EXPERIENCE_MIN,
                                        NUMEROSITY_MIN, TIME_STAMP_MIN)
from piecewise.error.classifier_error import AttrUpdateError

# make sure default numeric attr val is valid (prediction, error, fitness are
# reals and can be anything, but time_stamp is int and has a minimum value)
DEFAULT_NUMERIC_ATTR_VAL = TIME_STAMP_MIN
ALT_NUMERIC_ATTR_VAL = DEFAULT_NUMERIC_ATTR_VAL + 1


@pytest.fixture
def make_classifier(mocker):
    def _make_classifier(rule=None):
        if rule is None:
            rule = mocker.MagicMock()
        return Classifier(rule,
                          prediction=DEFAULT_NUMERIC_ATTR_VAL,
                          error=DEFAULT_NUMERIC_ATTR_VAL,
                          fitness=DEFAULT_NUMERIC_ATTR_VAL,
                          time_stamp=DEFAULT_NUMERIC_ATTR_VAL)

    return _make_classifier


@pytest.fixture
def classifier(make_classifier):
    return make_classifier()


class TestClassifier:
    def test_set_bad_time_stamp_below_min_val(self, classifier):
        with pytest.raises(AttrUpdateError):
            classifier.time_stamp = (TIME_STAMP_MIN - 1)

    def test_set_bad_time_stamp_non_integer(self, classifier):
        with pytest.raises(AttrUpdateError):
            classifier.time_stamp = float(TIME_STAMP_MIN)

    def test_set_bad_experience_below_min_val(self, classifier):
        with pytest.raises(AttrUpdateError):
            classifier.experience = (EXPERIENCE_MIN - 1)

    def test_set_bad_experience_non_integer(self, classifier):
        with pytest.raises(AttrUpdateError):
            classifier.experience = float(EXPERIENCE_MIN)

    def test_set_bad_action_set_size(self, classifier):
        with pytest.raises(AttrUpdateError):
            classifier.action_set_size = (ACTION_SET_SIZE_MIN - 1)

    def test_set_bad_numerosity_below_min_val(self, classifier):
        with pytest.raises(AttrUpdateError):
            classifier.numerosity = (NUMEROSITY_MIN - 1)

    def test_set_bad_numerosity_non_integer(self, classifier):
        with pytest.raises(AttrUpdateError):
            classifier.numerosity = float(NUMEROSITY_MIN)

    def test_is_microclassifier_pos_case(self, classifier):
        setattr(classifier, "numerosity", NUMEROSITY_MIN)
        assert classifier.is_micro and \
            not classifier.is_macro

    def test_is_macroclassiifer_pos_case(self, classifier):
        setattr(classifier, "numerosity", NUMEROSITY_MIN + 1)
        assert classifier.is_macro and \
            not classifier.is_micro

    @pytest.mark.parametrize("num_wildcards, expected_generality",
                             [(0, 0.0), (1, 50.0), (2, 100.0)])
    def test_generality_two_elem_condition(self, mocker, classifier,
                                           num_wildcards, expected_generality):
        condition_len = 2
        condition = mocker.MagicMock()
        condition.__len__.return_value = condition_len
        classifier._rule.condition = condition

        rule_repr = mocker.MagicMock()
        rule_repr.num_wildcards.return_value = num_wildcards

        assert classifier.generality_as_percentage(rule_repr) == \
            expected_generality

    @pytest.mark.parametrize("num_wildcards", [0, 1, 2])
    def test_num_wildcards(self, mocker, classifier, num_wildcards):
        rule_repr = mocker.MagicMock()
        rule_repr.num_wildcards.return_value = num_wildcards

        assert classifier.num_wildcards(rule_repr) == num_wildcards

    def test_eq_pos_case(self, make_classifier, mocker):
        shared_rule = mocker.MagicMock()
        classifier = make_classifier(rule=shared_rule)
        same_classifier = make_classifier(rule=shared_rule)
        assert classifier == same_classifier

    def test_ne_pos_case_diff_rule(self, make_classifier, mocker):
        classifier = make_classifier()
        diff_rule_classifier = make_classifier()
        assert classifier != diff_rule_classifier

    @pytest.mark.parametrize("attr_to_alter", [
        "prediction", "error", "fitness", "time_stamp", "experience",
        "action_set_size", "numerosity"
    ])
    def test_ne_pos_case_diff_numeric_attr(self, make_classifier,
                                           attr_to_alter):
        classifier = make_classifier()
        diff_numeric_attr_classifier = make_classifier()

        setattr(diff_numeric_attr_classifier, attr_to_alter,
                ALT_NUMERIC_ATTR_VAL)
        assert classifier != diff_numeric_attr_classifier

import pytest

import piecewise.dtype.classifier as classifier

MICRO_NUMEROSITY = classifier.NUMEROSITY_MIN


@pytest.fixture
def make_mock_microclassifier(mocker):
    def _make_mock_microclassifier():
        microclassifier = mocker.MagicMock()
        microclassifier.numerosity = MICRO_NUMEROSITY
        microclassifier.is_micro = True
        microclassifier.is_macro = False
        return microclassifier

    return _make_mock_microclassifier


@pytest.fixture
def mock_microclassifier(make_mock_microclassifier):
    return make_mock_microclassifier()


@pytest.fixture
def make_mock_macroclassifier(mocker):
    def _make_mock_macroclassifier(numerosity):
        macroclassifier = mocker.MagicMock()
        macroclassifier.numerosity = numerosity
        macroclassifier.is_micro = False
        macroclassifier.is_macro = True
        return macroclassifier

    return _make_mock_macroclassifier


@pytest.fixture
def make_mock_elem(mocker):
    def _make_mock_elem():
        return mocker.MagicMock()

    return _make_mock_elem


@pytest.fixture
def mock_elem(make_mock_elem):
    return make_mock_elem()

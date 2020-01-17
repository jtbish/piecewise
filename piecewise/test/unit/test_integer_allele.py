import pytest

from piecewise.dtype import IntegerAllele
from piecewise.error.allele_error import ConversionError


class TestIntegerAllele:
    @pytest.fixture
    def cannot_convert_to_int(self):
        return "string"

    def test_init_non_int_convertable_input(self, cannot_convert_to_int):
        with pytest.raises(ConversionError):
            IntegerAllele(cannot_convert_to_int)

    def test_eq_pos_case_against_same_type(self):
        zero_allele = IntegerAllele(0)
        another_zero_allele = IntegerAllele(0)
        assert zero_allele == another_zero_allele

    def test_eq_pos_case_against_int(self):
        zero_allele = IntegerAllele(0)
        assert zero_allele == 0

    def test_eq_non_int_convertable_input(self, cannot_convert_to_int):
        zero_allele = IntegerAllele(0)
        with pytest.raises(ConversionError):
            zero_allele == cannot_convert_to_int

    def test_ne_pos_case_against_same_type(self):
        zero_allele = IntegerAllele(0)
        one_allele = IntegerAllele(1)
        assert zero_allele != one_allele

    def test_ne_non_int_convertable_input(self, cannot_convert_to_int):
        zero_allele = IntegerAllele(0)
        with pytest.raises(ConversionError):
            zero_allele != cannot_convert_to_int

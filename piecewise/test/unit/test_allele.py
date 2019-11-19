import math

import numpy as np
import pytest

from piecewise.dtype import FloatAllele, IntegerAllele
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


class TestFloatAllele:
    _LESS_THAN_PI = 3
    _GREATER_THAN_PI = 4

    @pytest.fixture
    def pi_allele(self):
        return FloatAllele(math.pi)

    @pytest.fixture
    def approx_pi_allele(self):
        truncated_pi = 3.14159
        return FloatAllele(truncated_pi)

    @pytest.fixture
    def cannot_convert_to_float(self):
        return "string"

    def test_init_non_float_convertable_input(self, cannot_convert_to_float):
        with pytest.raises(ConversionError):
            FloatAllele(cannot_convert_to_float)

    def test_lt_pos_case(self, pi_allele):
        assert pi_allele < self._GREATER_THAN_PI

    def test_lt_neg_case_equal(self, pi_allele):
        assert not pi_allele < math.pi

    def test_lt_neg_case_greater(self, pi_allele):
        assert not pi_allele < self._LESS_THAN_PI

    def test_lt_non_float_convertable_input(self, pi_allele,
                                            cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele < cannot_convert_to_float

    def test_le_pos_case_less(self, pi_allele):
        assert pi_allele <= self._GREATER_THAN_PI

    def test_le_pos_case_equal(self, pi_allele):
        assert pi_allele <= math.pi

    def test_le_neg_case(self, pi_allele):
        assert not pi_allele <= self._LESS_THAN_PI

    def test_le_non_float_convertable_input(self, pi_allele,
                                            cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele <= cannot_convert_to_float

    def test_eq_pos_case_against_py_float(self, approx_pi_allele):
        assert approx_pi_allele == math.pi

    def test_eq_pos_case_against_np_float32(self, approx_pi_allele):
        assert approx_pi_allele == np.float32(np.pi)

    def test_eq_pos_case_against_np_float64(self, approx_pi_allele):
        assert approx_pi_allele == np.float64(np.pi)

    def test_eq_pos_case_against_same_type(self, approx_pi_allele, pi_allele):
        assert approx_pi_allele == pi_allele

    def test_eq_non_float_convertable_input(self, pi_allele,
                                            cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele == cannot_convert_to_float

    def test_ne_pos_case_against_same_type(self, pi_allele):
        euler_allele = FloatAllele(math.e)
        assert pi_allele != euler_allele

    def test_ne_non_float_convertable_input(self, pi_allele,
                                            cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele != cannot_convert_to_float

    def test_gt_pos_case(self, pi_allele):
        assert pi_allele > self._LESS_THAN_PI

    def test_gt_neg_case_less(self, pi_allele):
        assert not pi_allele > self._GREATER_THAN_PI

    def test_gt_neg_case_equal(self, pi_allele):
        assert not pi_allele > math.pi

    def test_gt_non_float_convertable_input(self, pi_allele,
                                            cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele > cannot_convert_to_float

    def test_ge_pos_case_greater(self, pi_allele):
        assert pi_allele >= self._LESS_THAN_PI

    def test_ge_pos_case_equal(self, pi_allele):
        assert pi_allele >= math.pi

    def test_ge_neg_case(self, pi_allele):
        assert not pi_allele >= self._GREATER_THAN_PI

    def test_ge_non_float_convertable_input(self, pi_allele,
                                            cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele >= cannot_convert_to_float

    def test_add_with_py_float(self):
        allele = FloatAllele(0.1)
        float_to_add = 0.2
        allele = allele + float_to_add
        assert allele == 0.3

    def test_add_with_same_type(self):
        allele = FloatAllele(0.1)
        allele_to_add = FloatAllele(0.2)
        allele = allele + allele_to_add
        assert allele == 0.3

    def test_add_non_float_convertable_input(self, pi_allele,
                                             cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele + cannot_convert_to_float

    def test_sub_with_py_float(self):
        allele = FloatAllele(0.3)
        float_to_sub = 0.2
        allele = allele - float_to_sub
        assert allele == 0.1

    def test_sub_with_same_type(self):
        allele = FloatAllele(0.3)
        allele_to_sub = FloatAllele(0.2)
        allele = allele - allele_to_sub
        assert allele == 0.1

    def test_sub_non_float_convertable_input(self, pi_allele,
                                             cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele - cannot_convert_to_float

    def test_iadd_with_py_float(self):
        allele = FloatAllele(0.1)
        float_to_add = 0.2
        allele += float_to_add
        assert allele == 0.3

    def test_iadd_with_same_type(self):
        allele = FloatAllele(0.1)
        allele_to_add = FloatAllele(0.2)
        allele += allele_to_add
        assert allele == 0.3

    def test_iadd_non_float_convertable_input(self, pi_allele,
                                              cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele += cannot_convert_to_float

    def test_isub_with_py_float(self):
        allele = FloatAllele(0.3)
        float_to_sub = 0.2
        allele -= float_to_sub
        assert allele == 0.1

    def test_isub_with_same_type(self):
        allele = FloatAllele(0.3)
        allele_to_sub = FloatAllele(0.2)
        allele -= allele_to_sub
        assert allele == 0.1

    def test_isub_non_float_convertable_input(self, pi_allele,
                                              cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele -= cannot_convert_to_float

import math

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from piecewise.dtype import FloatAllele
from piecewise.dtype.config import float_allele_rel_tol
from piecewise.error.allele_error import ConversionError

EQ_TESTING_DIFF = float_allele_rel_tol*2


class TestFloatAllele:
    _LESS_THAN_PI = 3
    _GREATER_THAN_PI = 4

    @pytest.fixture
    def pi_allele(self):
        return FloatAllele(math.pi)

    @pytest.fixture
    def approx_pi_allele(self):
        truncated_pi = math.pi - EQ_TESTING_DIFF
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

    @given(float1=st.floats(allow_nan=False, allow_infinity=False),
           float2=st.floats(allow_nan=False, allow_infinity=False))
    def test_add_with_py_float(self, float1, float2):
        allele = FloatAllele(float1)
        float_to_add = float2
        allele = allele + float_to_add
        assert allele == float1 + float2

    @given(float1=st.floats(allow_nan=False, allow_infinity=False),
           float2=st.floats(allow_nan=False, allow_infinity=False))
    def test_add_with_same_type(self, float1, float2):
        allele = FloatAllele(float1)
        allele_to_add = FloatAllele(float2)
        allele = allele + allele_to_add
        assert allele == float1 + float2

    def test_add_non_float_convertable_input(self, pi_allele,
                                             cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele + cannot_convert_to_float

    @given(float1=st.floats(allow_nan=False, allow_infinity=False),
           float2=st.floats(allow_nan=False, allow_infinity=False))
    def test_sub_with_py_float(self, float1, float2):
        allele = FloatAllele(float1)
        float_to_sub = float2
        allele = allele - float_to_sub
        assert allele == float1 - float2

    @given(float1=st.floats(allow_nan=False, allow_infinity=False),
           float2=st.floats(allow_nan=False, allow_infinity=False))
    def test_sub_with_same_type(self, float1, float2):
        allele = FloatAllele(float1)
        allele_to_sub = FloatAllele(float2)
        allele = allele - allele_to_sub
        assert allele == float1 - float2

    def test_sub_non_float_convertable_input(self, pi_allele,
                                             cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele - cannot_convert_to_float

    @given(float1=st.floats(allow_nan=False, allow_infinity=False),
           float2=st.floats(allow_nan=False, allow_infinity=False))
    def test_iadd_with_py_float(self, float1, float2):
        allele = FloatAllele(float1)
        float_to_add = float2
        allele += float_to_add
        assert allele == float1 + float2

    @given(float1=st.floats(allow_nan=False, allow_infinity=False),
           float2=st.floats(allow_nan=False, allow_infinity=False))
    def test_iadd_with_same_type(self, float1, float2):
        allele = FloatAllele(float1)
        allele_to_add = FloatAllele(float2)
        allele += allele_to_add
        assert allele == float1 + float2

    def test_iadd_non_float_convertable_input(self, pi_allele,
                                              cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele += cannot_convert_to_float

    @given(float1=st.floats(allow_nan=False, allow_infinity=False),
           float2=st.floats(allow_nan=False, allow_infinity=False))
    def test_isub_with_py_float(self, float1, float2):
        allele = FloatAllele(float1)
        float_to_sub = float2
        allele -= float_to_sub
        assert allele == float1 - float2

    @given(float1=st.floats(allow_nan=False, allow_infinity=False),
           float2=st.floats(allow_nan=False, allow_infinity=False))
    def test_isub_with_same_type(self, float1, float2):
        allele = FloatAllele(float1)
        allele_to_sub = FloatAllele(float2)
        allele -= allele_to_sub
        assert allele == float1 - float2

    def test_isub_non_float_convertable_input(self, pi_allele,
                                              cannot_convert_to_float):
        with pytest.raises(ConversionError):
            pi_allele -= cannot_convert_to_float

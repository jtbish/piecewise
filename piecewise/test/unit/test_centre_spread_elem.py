import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from piecewise.lcs.hyperparams import register_hyperparams
from piecewise.dtype import FloatAllele
from piecewise.dtype.config import float_allele_rel_tol
from piecewise.rule_repr.interval.elem.centre_spread_elem import \
    CentreSpreadElem

# set the minimum diff for centres and spreads to be clear of the float allele
# tolerance
MIN_DIFF = float_allele_rel_tol * 2
MIN_MUTATION_MAGNITUDE = MIN_DIFF

MIN_ALLELE_VALUE = -1.0
MAX_ALLELE_VALUE = 1.0
NUM_ALLELE_VALUES = 3
ALLELE_VALUES = list(
    np.random.uniform(low=MIN_ALLELE_VALUE,
                      high=MAX_ALLELE_VALUE,
                      size=NUM_ALLELE_VALUES))


class TestCentreSpreadElem:
    @pytest.fixture(params=ALLELE_VALUES)
    def centre_allele(self, request):
        allele_value = request.param
        return FloatAllele(allele_value)

    @pytest.fixture(params=ALLELE_VALUES)
    def spread_allele(self, request):
        allele_value = request.param
        return FloatAllele(allele_value)

    def test_eq_pos_case(self, centre_allele, spread_allele):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        same_elem = CentreSpreadElem(centre_allele, spread_allele)
        assert elem == same_elem

    @given(diff=st.floats(min_value=MIN_DIFF,
                          allow_nan=False,
                          allow_infinity=False))
    def test_ne_pos_case_diff_centre(self, centre_allele, spread_allele, diff):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        diff_centre_elem = CentreSpreadElem(centre_allele + diff,
                                            spread_allele)
        assert elem != diff_centre_elem

    @given(diff=st.floats(min_value=MIN_DIFF,
                          allow_nan=False,
                          allow_infinity=False))
    def test_ne_pos_case_diff_spread(self, centre_allele, spread_allele, diff):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        diff_spread_elem = CentreSpreadElem(centre_allele,
                                            spread_allele + diff)
        assert elem != diff_spread_elem

    @given(diff=st.floats(min_value=MIN_DIFF,
                          allow_nan=False,
                          allow_infinity=False))
    def test_ne_pos_case_both_diff(self, centre_allele, spread_allele, diff):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        both_diff_elem = \
            CentreSpreadElem(centre_allele + diff,
                             spread_allele + diff)
        assert elem != both_diff_elem

    def test_lower(self, centre_allele, spread_allele):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        expected_lower = centre_allele - spread_allele
        assert elem.lower() == expected_lower

    def test_upper(self, centre_allele, spread_allele):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        expected_upper = centre_allele + spread_allele
        assert elem.upper() == expected_upper

    @given(mutation_magnitude=st.floats(min_value=MIN_MUTATION_MAGNITUDE,
                                        allow_nan=False,
                                        allow_infinity=False))
    def test_mutate_force_enable(self, centre_allele, spread_allele,
                                 mutation_magnitude):
        orig_centre_value = centre_allele.value
        orig_spread_value = spread_allele.value
        elem = CentreSpreadElem(centre_allele, spread_allele)
        register_hyperparams({"mu": 1.0, "m": mutation_magnitude})
        elem.mutate()
        centre_value_valid = \
            (orig_centre_value - mutation_magnitude) <= elem._centre_allele \
            <= (orig_centre_value + mutation_magnitude)
        spread_value_valid = \
            (orig_spread_value - mutation_magnitude) <= elem._spread_allele \
            <= (orig_spread_value + mutation_magnitude)
        assert centre_value_valid and spread_value_valid

    @given(mutation_magnitude=st.floats(min_value=MIN_MUTATION_MAGNITUDE,
                                        allow_nan=False,
                                        allow_infinity=False))
    def test_mutate_force_disable(self, centre_allele, spread_allele,
                                  mutation_magnitude):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        register_hyperparams({"mu": 0.0, "m": mutation_magnitude})
        elem.mutate()
        centre_value_same = elem._centre_allele == centre_allele
        spread_value_same = elem._spread_allele == spread_allele
        assert centre_value_same and spread_value_same

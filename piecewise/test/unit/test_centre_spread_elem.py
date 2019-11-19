import pytest

from piecewise.dtype import FloatAllele
from piecewise.rule_repr.interval.elem.centre_spread_elem import \
    CentreSpreadElem

MUTATION_MAGNITUDE = 1.0


class TestCentreSpreadElem:
    _DIFF = 1.0

    @pytest.fixture(params=[-1.0, 0.0, 1.0])
    def centre_allele(self, request):
        allele_value = request.param
        return FloatAllele(allele_value)

    @pytest.fixture(params=[-1.0, 0.0, 1.0])
    def spread_allele(self, request):
        allele_value = request.param
        return FloatAllele(allele_value)

    def test_eq_pos_case(self, centre_allele, spread_allele):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        same_elem = CentreSpreadElem(centre_allele, spread_allele)
        assert elem == same_elem

    def test_ne_pos_case_diff_centre(self, centre_allele, spread_allele):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        diff_centre_elem = CentreSpreadElem(centre_allele + self._DIFF,
                                            spread_allele)
        assert elem != diff_centre_elem

    def test_ne_pos_case_diff_spread(self, centre_allele, spread_allele):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        diff_spread_elem = CentreSpreadElem(centre_allele,
                                            spread_allele + self._DIFF)
        assert elem != diff_spread_elem

    def test_ne_pos_case_both_diff(self, centre_allele, spread_allele):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        both_diff_elem = \
            CentreSpreadElem(centre_allele + self._DIFF,
                             spread_allele + self._DIFF)
        assert elem != both_diff_elem

    def test_lower(self, centre_allele, spread_allele):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        expected_lower = centre_allele - spread_allele
        assert elem.lower() == expected_lower

    def test_upper(self, centre_allele, spread_allele):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        expected_upper = centre_allele + spread_allele
        assert elem.upper() == expected_upper

    def test_mutate_force_enable(self, centre_allele, spread_allele):
        orig_centre_value = centre_allele.value
        orig_spread_value = spread_allele.value
        elem = CentreSpreadElem(centre_allele, spread_allele)
        hyperparams = {"mu": 1.0, "m": MUTATION_MAGNITUDE}
        elem.mutate(hyperparams)
        centre_value_valid = \
            (orig_centre_value - MUTATION_MAGNITUDE) <= elem._centre_allele \
            <= (orig_centre_value + MUTATION_MAGNITUDE)
        spread_value_valid = \
            (orig_spread_value - MUTATION_MAGNITUDE) <= elem._spread_allele \
            <= (orig_spread_value + MUTATION_MAGNITUDE)
        assert centre_value_valid and spread_value_valid

    def test_mutate_force_disable(self, centre_allele, spread_allele):
        elem = CentreSpreadElem(centre_allele, spread_allele)
        hyperparams = {"mu": 0.0, "m": MUTATION_MAGNITUDE}
        elem.mutate(hyperparams)
        centre_value_same = elem._centre_allele == centre_allele
        spread_value_same = elem._spread_allele == spread_allele
        assert centre_value_same and spread_value_same

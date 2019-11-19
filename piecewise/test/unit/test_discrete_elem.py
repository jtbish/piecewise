from piecewise.dtype import IntegerAllele
from piecewise.rule_repr.discrete.elem.discrete_elem import (
    DiscreteElem, DiscreteWildcardElem)


class TestDiscreteElem:
    def test_eq_pos_case_against_same_type(self):
        allele_val = 0
        elem = DiscreteElem(IntegerAllele(allele_val))
        same_elem = DiscreteElem(IntegerAllele(allele_val))
        assert elem == same_elem

    def test_eq_pos_case_against_int(self):
        allele_val = 0
        elem = DiscreteElem(IntegerAllele(allele_val))
        int_ = allele_val
        assert elem == int_

    def test_ne_pos_case_against_same_type(self):
        allele_val = 0
        elem = DiscreteElem(IntegerAllele(allele_val))
        diff_allele_val = allele_val + 1
        diff_elem = DiscreteElem(IntegerAllele(diff_allele_val))
        assert elem != diff_elem

    def test_ne_pos_case_against_int(self):
        allele_val = 0
        elem = DiscreteElem(IntegerAllele(allele_val))
        diff_int = allele_val + 1
        assert elem != diff_int

    def test_ne_pos_case_against_wildcard(self):
        allele_val = 0
        elem = DiscreteElem(IntegerAllele(allele_val))
        wildcard_elem = DiscreteWildcardElem()
        assert elem != wildcard_elem

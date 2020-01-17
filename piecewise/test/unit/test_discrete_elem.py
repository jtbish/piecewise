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


class TestDiscreteWildcardElem:
    def test_eq_pos_case_against_same_type(self):
        elem1 = DiscreteWildcardElem()
        elem2 = DiscreteWildcardElem()
        assert elem1 == elem2

    def test_ne_pos_case_against_discrete_elem(self):
        allele_val = 0
        discrete_elem = DiscreteElem(IntegerAllele(allele_val))
        wildcard_elem = DiscreteWildcardElem()
        assert wildcard_elem != discrete_elem

    def test_ne_pos_case_against_other_type(self):
        wildcard_elem = DiscreteWildcardElem()
        assert wildcard_elem != "string"


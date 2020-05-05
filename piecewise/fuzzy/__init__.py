from .covering import (FuzzyRuleReprCovering,
                       make_fuzzy_linear_prediction_classifier)
from .credit_assignment import FuzzyXCSFLinearPredictionCreditAssignment
from .domain import Domain
from .linguistic_var import LinguisticVar
from .logical_ops import logical_and_min, logical_or_max
from .membership_func import (make_trapezoidal_membership_func,
                              make_triangular_membership_func)
from .prediction import FuzzyMatchingFitnessWeightedAvgPrediction
from .rule_repr import FuzzyMinSpanRuleRepr, FuzzyConjunctiveRuleRepr

from .action_selection import FixedEpsilonGreedy, LinearDecayEpsilonGreedy
from .covering import (RuleReprCovering, make_classifier,
                       make_linear_prediction_classifier)
from .credit_assignment import (XCSCreditAssignment,
                                XCSFLinearPredictionCreditAssignment)
from .deletion import XCSRouletteWheelDeletion
from .fitness_update import XCSAccuracyFitnessUpdate
from .matching import RuleReprMatching
from .prediction import FitnessWeightedAvgPrediction
from .rule_discovery.ga.xcs_genetic_algorithm import (make_canonical_xcs_ga,
                                                      make_improved_xcs_ga,
                                                      make_custom_xcs_ga)
from .subsumption import XCSSubsumption

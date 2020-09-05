from .action_selection import (ExpDecayEpsilonGreedy, FixedEpsilonGreedy,
                               LinearDecayEpsilonGreedy)
from .covering import (RuleReprCovering, make_classifier,
                       make_linear_prediction_classifier,
                       NullCovering)
from .credit_assignment import (XCSCreditAssignment,
                                XCSFLinearPredictionCreditAssignment)
from .deletion import XCSRouletteWheelDeletion, NullDeletion
from .fitness_update import XCSAccuracyFitnessUpdate, NullFitnessUpdate
from .matching import RuleReprMatching
from .prediction import FitnessWeightedAvgPrediction
from .rule_discovery.rule_discovery import NullRuleDiscovery
from .rule_discovery.ga.xcs_genetic_algorithm import (make_canonical_xcs_ga,
                                                      make_custom_xcs_ga,
                                                      make_improved_xcs_ga)
from .subsumption import XCSSubsumption, NullSubsumption

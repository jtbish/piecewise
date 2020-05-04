from piecewise.dtype import Rule
from piecewise.lcs.component.covering import RuleReprCoveringABC


class FuzzyRuleReprCovering(RuleReprCoveringABC):
    def __call__(self, population, match_set, situation, time_step):
        clfrs_with_max_matching_degree = \
            self._find_classifiers_with_max_matching_degree(match_set,
                                                            situation)
        represented_actions = \
            self._find_represented_actions(clfrs_with_max_matching_degree)
        unrepresented_actions = self._env_action_set - represented_actions
        self._cover_unrepresented_actions(population, match_set, situation,
                                          time_step, unrepresented_actions)

    def _find_classifiers_with_max_matching_degree(self, match_set, situation):
        max_matching_degree = \
            self._rule_repr.calc_max_matching_degree(situation)
        return [
            classifier for classifier in match_set
            if classifier.matching_degree == max_matching_degree
        ]

    def _find_represented_actions(self, classifiers):
        represented_actions = set()
        for classifier in classifiers:
            represented_actions.add(classifier.action)
        return represented_actions

    def _cover_unrepresented_actions(self, population, match_set, situation,
                                     time_step, unrepresented_actions):
        for action in unrepresented_actions:
            covering_condition = \
                self._rule_repr.gen_covering_condition(situation)
            covering_action = action
            rule = Rule(covering_condition,
                        covering_action,
                        num_features=len(situation))
            classifier = self._classifier_factory(rule, time_step)
            population.add(classifier, operation_label="covering")
            match_set.add(classifier)
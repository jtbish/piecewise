import abc

from piecewise.component import (EpsilonGreedy, FitnessWeightedAvgPrediction,
                                 RuleReprCovering, RuleReprMatching,
                                 XCSAccuracyFitnessUpdate, XCSCreditAssignment,
                                 XCSGeneticAlgorithm, XCSRouletteWheelDeletion,
                                 XCSSubsumption)
from piecewise.environment import EnvironmentStepTypes
from piecewise.util.classifier_set_stats import (calc_summary_stat,
                                                 num_unique_actions)

from .algorithm import AlgorithmABC, AlgorithmComponents


def make_xcs(env_step_type, *args, **kwargs):
    """Factory function to make suitable XCS instance given type of
    environment."""
    if env_step_type == EnvironmentStepTypes.single_step:
        return SingleStepXCS(*args, **kwargs)
    elif env_step_type == EnvironmentStepTypes.multi_step:
        return MultiStepXCS(*args, **kwargs)
    else:
        # TODO change
        raise Exception


class XCS(AlgorithmABC, metaclass=abc.ABCMeta):
    """Implementation of XCS, based on pseudocode given in 'An Algorithmic
    Description of XCS' (Butz and Wilson, 2002)."""
    def __init__(self, env_action_set, rule_repr, hyperparams):
        components = self._init_components(env_action_set, rule_repr,
                                           hyperparams)
        super().__init__(components, rule_repr, hyperparams)

        self._init_prev_step_tracking_attrs()
        self._init_curr_step_tracking_attrs()

    def _init_components(self, env_action_set, rule_repr, hyperparams):
        matching_strat = RuleReprMatching(rule_repr)
        covering_strat = RuleReprCovering(env_action_set, rule_repr,
                                          hyperparams)
        prediction_strat = FitnessWeightedAvgPrediction(env_action_set)
        fitness_update_strat = \
            XCSAccuracyFitnessUpdate(hyperparams)
        subsumption_strat = XCSSubsumption(rule_repr, hyperparams)
        ga_subsumption_strat = subsumption_strat if \
            hyperparams["do_ga_subsumption"] \
            else None
        rule_discovery_strat = XCSGeneticAlgorithm(env_action_set, rule_repr,
                                                   ga_subsumption_strat,
                                                   hyperparams)
        deletion_strat = XCSRouletteWheelDeletion(hyperparams)
        action_selection_strat = \
            EpsilonGreedy(hyperparams["p_explore"])
        credit_assignment_strat = \
            XCSCreditAssignment(hyperparams)

        return AlgorithmComponents(matching_strat, covering_strat,
                                   prediction_strat, fitness_update_strat,
                                   subsumption_strat, rule_discovery_strat,
                                   deletion_strat, action_selection_strat,
                                   credit_assignment_strat)

    def _init_prev_step_tracking_attrs(self):
        self._prev_action_set = None
        self._prev_reward = None
        self._prev_situation = None

    def _init_curr_step_tracking_attrs(self):
        self._action_set = None
        self._prediction_array = None
        self._situation = None
        self._time_step = None

    def train_query(self, situation, time_step):
        """First half (until line 7) of RUN EXPERIMENT function from
        'An Algorithmic Description of XCS' (Butz and Wilson, 2002).

        Only represents a single iteration of the do-while loop in
        RUN EXPERIMENT, as the caller controls termination criteria."""
        self._situation = situation
        self._time_step = time_step
        match_set = self._gen_match_set(self._situation)
        self._perform_covering(match_set)
        self._prediction_array = self._gen_prediction_array(match_set)
        action = self._select_action(self._prediction_array)
        self._action_set = self._gen_action_set(match_set, action)

        return action

    def _perform_covering(self, match_set):
        """Second loop of GENERATE MATCH SET function from
        'An Algorithmic Description of XCS' (Butz and Wilson, 2002)."""
        while self._should_cover(match_set):
            covering_classifier = self._gen_covering_classifier(
                match_set, self._situation, self._time_step)
            self._population.add(covering_classifier,
                                 operation_label="covering")
            match_set.add(covering_classifier)

    def _should_cover(self, match_set):
        return num_unique_actions(match_set) < self._hyperparams["theta_mna"]

    def train_update(self, env_response):
        """Second half (line 8 onwards) of RUN EXPERIMENT function from
        'An Algorithmic Description of XCS' (Butz and Wilson, 2002).

        Only represents a single iteration of the do-while loop in
        RUN EXPERIMENT, as the caller controls termination criteria.

        Update steps to follow are different based on whether XCS is being
        applied to a single-step or multi-step problem."""
        self._step_type_train_update(env_response)
        self._perform_deletion()
        return self._population

    @abc.abstractmethod
    def _step_type_train_update(self, env_response):
        raise NotImplementedError

    def _update_curr_action_set(self, reward):
        self._update_action_set(self._action_set,
                                self._situation,
                                reward,
                                use_discounting=False,
                                prediction_array=None)
        self._previous_action_set = None

    def _update_action_set(self,
                           action_set,
                           situation,
                           reward,
                           *,
                           use_discounting=False,
                           prediction_array=None):
        """Performs updates in the given action set (either the previous action
        set or the current action set).

        These updates are composed of credit assignment, fitness updates,
        possible subsumption, and rule discovery. This method primarily
        resembles the UPDATE SET function from 'An Algorithmic Description of
        XCS' (Butz and Wilson, 2002), with the addition of running the rule
        discovery step last."""
        self._do_credit_assignment(action_set, reward, use_discounting,
                                   prediction_array)
        self._update_fitness(action_set)
        if self._hyperparams["do_as_subsumption"]:
            self._do_action_set_subsumption(action_set)
        if self._should_do_rule_discovery():
            self._discover_classifiers(action_set, self._population,
                                       self._situation, self._time_step)

    def _do_action_set_subsumption(self, action_set):
        """DO ACTION SET SUBSUMPTION function from 'An Algorithmic Description
        of XCS' (Butz and Wilson, 2002).
        """
        most_general_classifier = \
            self._find_most_general_classifier(action_set)
        self._perform_subsumptions(most_general_classifier, action_set)

    def _find_most_general_classifier(self, action_set):
        most_general_classifier = None
        for classifier in action_set:
            if self._subsumption_strat.could_subsume(classifier):
                if most_general_classifier is None or \
                        self._subsumption_strat.is_more_general(
                            classifier, most_general_classifier):
                    most_general_classifier = classifier
        return most_general_classifier

    def _perform_subsumptions(self, most_general_classifier, action_set):
        if most_general_classifier is not None:
            for classifier in action_set:
                if self._subsumption_strat.is_more_general(
                        most_general_classifier, classifier):
                    action_set.remove(classifier)
                    self._population.replace(replacee=classifier,
                                             replacer=most_general_classifier,
                                             operation_label="as_subsumption")

    def _should_do_rule_discovery(self):
        mean_time_stamp_in_pop = calc_summary_stat(self._population, "mean",
                                                   "time_stamp")
        time_since_last_rule_discovery = self._time_step - \
            mean_time_stamp_in_pop
        return time_since_last_rule_discovery > \
            self._hyperparams["theta_ga"]

    def test_query(self, situation):
        match_set = self._gen_match_set(situation)
        prediction_array = self._gen_prediction_array(match_set)
        return prediction_array.greedy_action()


class SingleStepXCS(XCS):
    """Training update strategy for single-step environments."""
    def _step_type_train_update(self, env_response):
        assert self._prev_action_set is None
        reward = env_response.reward
        self._update_curr_action_set(reward)


class MultiStepXCS(XCS):
    """Training update strategy for multi-step environments."""
    def _step_type_train_update(self, env_response):
        self._try_update_prev_action_set()
        reward = env_response.reward
        env_is_terminal = env_response.is_terminal
        if env_is_terminal:
            self._update_curr_action_set(reward)
        else:
            self._prev_action_set = self._action_set
            self._prev_reward = reward
            self._prev_situation = self._situation

    def _try_update_prev_action_set(self):
        if self._prev_action_set is not None:
            assert self._prev_situation is not None
            assert self._prev_reward is not None
            self._update_action_set(self._prev_action_set,
                                    self._prev_situation,
                                    self._prev_reward,
                                    use_discounting=True,
                                    prediction_array=self._prediction_array)

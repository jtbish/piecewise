import abc
import copy
import logging
from collections import namedtuple

from piecewise.dtype import ClassifierSet
from piecewise.environment import EnvironmentStepTypes
from piecewise.error.classifier_set_error import MemberNotFoundError
from piecewise.error.core_errors import InternalError
from piecewise.util.classifier_set_stats import (calc_summary_stat,
                                                 num_unique_actions)

from .component import (FitnessWeightedAvgPrediction, FixedEpsilonGreedy,
                        RuleReprCovering, RuleReprMatching,
                        XCSAccuracyFitnessUpdate, XCSCreditAssignment,
                        XCSFLinearPredictionCreditAssignment,
                        XCSRouletteWheelDeletion, XCSSubsumption,
                        make_canonical_xcs_ga, make_classifier,
                        make_linear_prediction_classifier)
from .component.action_selection import select_greedy_action
from .hyperparams import get_hyperparam
from .lcs import LCS, LCSTrainResponse

XCSComponents = namedtuple("XCSComponents", [
    "matching", "covering", "prediction", "action_selection",
    "credit_assignment", "fitness_update", "subsumption", "rule_discovery",
    "deletion"
])


def make_canonical_xcs(env, rule_repr, hyperparams, seed):
    """Public factory function to make instance of 'Canonical XCS' for the
    given environment and rule repr, i.e. XCS with components as described in
    'An Algorithmic Description of XCS' (Butz and Wilson, 2002)'."""
    matching = RuleReprMatching(rule_repr)
    covering = RuleReprCovering(env.action_set,
                                rule_repr,
                                classifier_factory=make_classifier)
    prediction = FitnessWeightedAvgPrediction(env.action_set)
    action_selection = FixedEpsilonGreedy()
    credit_assignment = XCSCreditAssignment()
    fitness_update = XCSAccuracyFitnessUpdate()
    subsumption = XCSSubsumption(rule_repr)
    rule_discovery = make_canonical_xcs_ga(env.action_set, rule_repr,
                                           subsumption)
    deletion = XCSRouletteWheelDeletion()

    components = XCSComponents(matching, covering, prediction,
                               action_selection, credit_assignment,
                               fitness_update, subsumption, rule_discovery,
                               deletion)
    return _make_xcs(env.step_type, components, rule_repr, hyperparams, seed)


def make_custom_xcs(env, matching, covering, prediction, action_selection,
                    credit_assignment, fitness_update, subsumption,
                    rule_discovery, deletion, rule_repr, hyperparams, seed):
    """Public factory function to make instance of XCS with custom
    components."""
    components = XCSComponents(matching, covering, prediction,
                               action_selection, credit_assignment,
                               fitness_update, subsumption, rule_discovery,
                               deletion)
    return _make_xcs(env.step_type, components, rule_repr, hyperparams, seed)


def make_custom_xcs_from_canonical_base(env,
                                        rule_repr,
                                        hyperparams,
                                        seed,
                                        matching=None,
                                        covering=None,
                                        prediction=None,
                                        action_selection=None,
                                        credit_assignment=None,
                                        fitness_update=None,
                                        subsumption=None,
                                        rule_discovery=None,
                                        deletion=None):
    if matching is None:
        matching = RuleReprMatching(rule_repr)
    if covering is None:
        covering = RuleReprCovering(env.action_set,
                                    rule_repr,
                                    classifier_factory=make_classifier)
    if prediction is None:
        prediction = FitnessWeightedAvgPrediction(env.action_set)
    if action_selection is None:
        action_selection = FixedEpsilonGreedy()
    if credit_assignment is None:
        credit_assignment = XCSCreditAssignment()
    if fitness_update is None:
        fitness_update = XCSAccuracyFitnessUpdate()
    if subsumption is None:
        subsumption = XCSSubsumption(rule_repr)
    if rule_discovery is None:
        rule_discovery = make_canonical_xcs_ga(env.action_set, rule_repr,
                                               subsumption)
    if deletion is None:
        deletion = XCSRouletteWheelDeletion()

    components = XCSComponents(matching, covering, prediction,
                               action_selection, credit_assignment,
                               fitness_update, subsumption, rule_discovery,
                               deletion)
    return _make_xcs(env.step_type, components, rule_repr, hyperparams, seed)


def make_custom_xcsf_from_canonical_base(env,
                                         rule_repr,
                                         hyperparams,
                                         seed,
                                         matching=None,
                                         covering=None,
                                         prediction=None,
                                         action_selection=None,
                                         credit_assignment=None,
                                         fitness_update=None,
                                         subsumption=None,
                                         rule_discovery=None,
                                         deletion=None):
    if matching is None:
        matching = RuleReprMatching(rule_repr)
    if covering is None:
        covering = RuleReprCovering(
            env.action_set,
            rule_repr,
            classifier_factory=make_linear_prediction_classifier)
    if prediction is None:
        prediction = FitnessWeightedAvgPrediction(env.action_set)
    if action_selection is None:
        action_selection = FixedEpsilonGreedy()
    if credit_assignment is None:
        credit_assignment = XCSFLinearPredictionCreditAssignment()
    if fitness_update is None:
        fitness_update = XCSAccuracyFitnessUpdate()
    if subsumption is None:
        subsumption = XCSSubsumption(rule_repr)
    if rule_discovery is None:
        rule_discovery = make_canonical_xcs_ga(env.action_set, rule_repr,
                                               subsumption)
    if deletion is None:
        deletion = XCSRouletteWheelDeletion()

    components = XCSComponents(matching, covering, prediction,
                               action_selection, credit_assignment,
                               fitness_update, subsumption, rule_discovery,
                               deletion)
    return _make_xcs(env.step_type, components, rule_repr, hyperparams, seed)


def _make_xcs(env_step_type, *args, **kwargs):
    """Private factory function to make suitable XCS instance given type of
    environment."""
    if env_step_type == EnvironmentStepTypes.single_step:
        return SingleStepXCS(*args, **kwargs)
    elif env_step_type == EnvironmentStepTypes.multi_step:
        return MultiStepXCS(*args, **kwargs)
    else:
        raise InternalError(f"Invalid environment step type: {env_step_type}")


class XCSABC(LCS, metaclass=abc.ABCMeta):
    """Implementation of XCS, based on pseudocode given in 'An Algorithmic
    Description of XCS' (Butz and Wilson, 2002)."""
    def __init__(self, components, rule_repr, hyperparams, seed):
        super().__init__(rule_repr, hyperparams, seed)
        self._init_component_strats(components)
        self._init_prev_step_tracking_attrs()
        self._init_curr_step_tracking_attrs()

    def _init_component_strats(self, components):
        self._matching_strat = components.matching
        self._covering_strat = components.covering
        self._prediction_strat = components.prediction
        self._action_selection_strat = components.action_selection
        self._credit_assignment_strat = components.credit_assignment
        self._fitness_update_strat = components.fitness_update
        self._subsumption_strat = components.subsumption
        self._rule_discovery_strat = components.rule_discovery
        self._deletion_strat = components.deletion

    def _init_prev_step_tracking_attrs(self):
        self._prev_action_set = None
        self._prev_reward = None
        self._prev_situation = None

    def _init_curr_step_tracking_attrs(self):
        self._match_set = None
        self._action_set = None
        self._prediction_array = None
        self._situation = None
        self._time_step = None
        self._did_explore = None

    def train_query(self, situation, time_step):
        """First half (until line 7) of RUN EXPERIMENT function from
        'An Algorithmic Description of XCS' (Butz and Wilson, 2002).

        Only represents a single iteration of the do-while loop in
        RUN EXPERIMENT, as the caller controls termination criteria."""
        self._situation = situation
        self._time_step = time_step
        self._match_set = self._gen_match_set(self._situation)
        self._perform_covering(self._match_set)
        self._prediction_array = \
            self._gen_prediction_array(self._match_set, self._situation)
        action_select_response = \
            self._select_action(self._prediction_array, self._time_step)
        action = action_select_response.action
        self._action_set = self._gen_action_set(self._match_set, action)
        self._did_explore = action_select_response.did_explore

        return LCSTrainResponse(action=action, did_explore=self._did_explore)

    def gen_match_set(self, situation):
        return self._matching_strat(self._population, situation)

    def gen_action_set(self, match_set, action):
        """GENERATE ACTION SET function from 'An Algorithmic
        Description of XCS' (Butz and Wilson, 2002)."""
        action_set = ClassifierSet()
        for classifier in match_set:
            if classifier.action == action:
                action_set.add(classifier)
        return action_set

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
        return num_unique_actions(match_set) < get_hyperparam("theta_mna")

    def gen_prediction_array(self, match_set, situation=None):
        return self._prediction_strat(match_set, situation)

    def train_update(self, env_response):
        """Second half (line 8 onwards) of RUN EXPERIMENT function from
        'An Algorithmic Description of XCS' (Butz and Wilson, 2002).

        Only represents a single iteration of the do-while loop in
        RUN EXPERIMENT, as the caller controls termination criteria.

        Update steps to follow are different based on whether XCS is being
        applied to a single-step or multi-step problem."""
        self._step_type_train_update(env_response)
        self._perform_deletion()

    @abc.abstractmethod
    def _step_type_train_update(self, env_response):
        """Update specific for single/multi step environment - implemented in
        XCS subclasses."""
        raise NotImplementedError

    def _update_action_set(self, action_set, situation, payoff):
        """Performs updates in the given action set (either the previous action
        set or the current action set).

        These updates are composed of credit assignment, fitness updates,
        possible subsumption, and rule discovery. This method primarily
        resembles the UPDATE SET function from 'An Algorithmic Description of
        XCS' (Butz and Wilson, 2002), with the addition of running the rule
        discovery step last."""
        self._do_credit_assignment(action_set, payoff, situation)
        self._update_fitness(action_set)
        if get_hyperparam("do_as_subsumption"):
            self._do_action_set_subsumption(action_set)
        if self._should_do_rule_discovery_in_action_set(action_set):
            self._discover_classifiers(action_set, self._population, situation,
                                       self._time_step)

    def _do_action_set_subsumption(self, action_set):
        """DO ACTION SET SUBSUMPTION function from 'An Algorithmic Description
        of XCS' (Butz and Wilson, 2002).
        """
        most_general_classifier = \
            self._find_most_general_classifier_in_action_set(action_set)
        self._perform_action_set_subsumptions(most_general_classifier,
                                              action_set)

    def _find_most_general_classifier_in_action_set(self, action_set):
        most_general_classifier = None
        for classifier in action_set:
            if self._subsumption_strat.could_subsume(classifier):
                if most_general_classifier is None or \
                        self._subsumption_strat.is_more_general(
                            classifier, most_general_classifier):
                    most_general_classifier = classifier
        return most_general_classifier

    def _perform_action_set_subsumptions(self, most_general_classifier,
                                         action_set):
        if most_general_classifier is not None:
            # loop over copy of action set because possibly removing
            # classifiers from it during loop
            action_set_copy = copy.deepcopy(action_set)
            for classifier in action_set_copy:
                if self._subsumption_strat.is_more_general(
                        most_general_classifier, classifier):
                    logging.debug("Attempting to do an action set "
                                  "subsumption.")
                    logging.debug(f"Subsumer: {most_general_classifier}")
                    logging.debug(f"Subsumee: {classifier}")
                    action_set.remove(classifier)
                    self._try_subsume_in_population(
                        replacee=classifier, replacer=most_general_classifier)

    def _try_subsume_in_population(self, replacee, replacer):
        try:
            self._population.replace(replacee,
                                     replacer,
                                     operation_label="as_subsumption")
        except MemberNotFoundError:
            logging.debug("AS subsumption failure.")
        else:
            logging.debug("AS subsumption success.")

    def _should_do_rule_discovery_in_action_set(self, action_set):
        mean_time_stamp_in_action_set = calc_summary_stat(
            action_set, "mean", "time_stamp")
        time_since_last_rule_discovery = self._time_step - \
            mean_time_stamp_in_action_set
        return time_since_last_rule_discovery > \
            get_hyperparam("theta_ga") and self._did_explore

    def test_query(self, situation):
        match_set = self._gen_match_set(situation)
        prediction_array = self._gen_prediction_array(match_set)
        return select_greedy_action(prediction_array)

    # Private forwarding functions that do logging calls if needed
    def _gen_match_set(self, situation):
        match_set = self.gen_match_set(situation)
        logging.debug("Match set")
        logging.debug(f"{match_set}")
        return match_set

    def _gen_covering_classifier(self, match_set, situation, time_step):
        logging.info("Generating covering classifier.")
        classifier = self._covering_strat(match_set, situation, time_step)
        logging.debug(f"{classifier}")
        return classifier

    def _gen_prediction_array(self, match_set, situation):
        prediction_array = self.gen_prediction_array(match_set, situation)
        logging.debug(f"Prediction array: {prediction_array}")
        return prediction_array

    def _gen_action_set(self, match_set, action):
        action_set = self.gen_action_set(match_set, action)
        logging.debug("Action set")
        logging.debug(f"{action_set}")
        return action_set

    def _update_fitness(self, operating_set):
        logging.debug("Updating fitness")
        self._fitness_update_strat(operating_set)

    def _discover_classifiers(self, operating_set, population, situation,
                              time_step):
        logging.debug("Doing rule discovery")
        return self._rule_discovery_strat(operating_set, population, situation,
                                          time_step)

    def _perform_deletion(self):
        self._deletion_strat(self._population)

    def _select_action(self, prediction_array, time_step):
        action_select_res = self._action_selection_strat(
            prediction_array, time_step)
        action = action_select_res.action
        if action_select_res.did_explore:
            logging.debug(f"Action selection: explored, {action}")
        else:
            logging.debug(f"Action selection: exploited, {action}")
        return action_select_res

    def _do_credit_assignment(self, action_set, payoff, situation):
        logging.debug("Doing credit assignment.")
        self._credit_assignment_strat(action_set, payoff, situation)


class SingleStepXCS(XCSABC):
    """XCS operating in single-step environments."""
    def _step_type_train_update(self, env_response):
        self._assert_prev_step_tracking_attrs_are_null()
        # perform updates only in [A] using immediate reward as payoff
        payoff = env_response.reward
        self._update_action_set(self._action_set, self._situation, payoff)

    def _assert_prev_step_tracking_attrs_are_null(self):
        assert self._prev_action_set is None
        assert self._prev_reward is None
        assert self._prev_situation is None


class MultiStepXCS(XCSABC):
    """XCS operating in multi-step environments."""
    def _step_type_train_update(self, env_response):
        self._try_update_prev_action_set()
        self._try_update_curr_action_set(env_response)

    def _try_update_prev_action_set(self):
        if self._prev_action_set is not None:
            logging.debug("Updating prev as")
            assert self._prev_situation is not None
            assert self._prev_reward is not None
            payoff = self._calc_discounted_payoff()
            self._update_action_set(self._prev_action_set,
                                    self._prev_situation, payoff)

    def _calc_discounted_payoff(self):
        max_prediction = max(self._prediction_array.values())
        payoff = self._prev_reward + (get_hyperparam("gamma") * max_prediction)
        return payoff

    def _try_update_curr_action_set(self, env_response):
        reward = env_response.reward
        if env_response.is_terminal:
            logging.debug("Terminal state, updating curr as")
            payoff = reward
            self._update_action_set(self._action_set, self._situation, payoff)
            self._prev_action_set = None
        else:
            logging.debug("Non-terminal state, rebinding prev as var")
            self._update_prev_step_tracking_attrs(reward)

    def _update_prev_step_tracking_attrs(self, reward):
        self._prev_action_set = self._action_set
        self._prev_reward = reward
        self._prev_situation = self._situation

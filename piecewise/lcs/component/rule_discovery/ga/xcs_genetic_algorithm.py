import copy
import logging
from collections import namedtuple

from piecewise.error.classifier_set_error import MemberNotFoundError
from piecewise.lcs.hyperparams import get_hyperparam
from piecewise.lcs.rng import get_rng

from .operator.crossover import TwoPointCrossover, UniformCrossover
from .operator.mutation import RuleReprMutation
from .operator.selection import RouletteWheelSelection, TournamentSelection

GAOperators = namedtuple("GAOperators", ["selection", "crossover", "mutation"])
ClassifierPair = namedtuple("ClassifierPair", ["first", "second"])


def make_canonical_xcs_ga(env_action_set, rule_repr, subsumption):
    selection = RouletteWheelSelection()
    crossover = TwoPointCrossover()
    mutation = RuleReprMutation(env_action_set, rule_repr)
    ga_operators = GAOperators(selection, crossover, mutation)
    return XCSGeneticAlgorithm(env_action_set, rule_repr, subsumption,
                               ga_operators)


def make_improved_xcs_ga(env_action_set, rule_repr, subsumption):
    selection = TournamentSelection()
    crossover = UniformCrossover()
    mutation = RuleReprMutation(env_action_set, rule_repr)
    ga_operators = GAOperators(selection, crossover, mutation)
    return XCSGeneticAlgorithm(env_action_set, rule_repr, subsumption,
                               ga_operators)


def make_custom_xcs_ga(env_action_set, rule_repr, subsumption, selection,
                       crossover, mutation):
    ga_operators = GAOperators(selection, crossover, mutation)
    return XCSGeneticAlgorithm(env_action_set, rule_repr, subsumption,
                               ga_operators)


class XCSGeneticAlgorithm:
    def __init__(self, env_action_set, rule_repr, subsumption, ga_operators):
        self._env_action_set = env_action_set
        self._rule_repr = rule_repr
        self._subsumption_strat = subsumption
        (self._selection_strat, self._crossover_strat,
         self._mutation_strat) = ga_operators

    def __call__(self, action_set, population, situation, time_step):
        """RUN GA function from 'An Algorithmic Description of
        XCS' (Butz and Wilson, 2002), without initial time check - which is
        factored into the caller.
        """
        self._update_participant_time_stamps(action_set, time_step)
        parents, children = self._select_parents_and_init_children(action_set)
        self._perform_crossover(children, parents, situation)
        self._perform_mutation(children, situation)
        self._update_population(children, parents, population)

    def _update_participant_time_stamps(self, action_set, time_step):
        for classifier in action_set:
            classifier.time_stamp = time_step

    def _select_parents_and_init_children(self, action_set):
        parent_one = self._selection_strat(action_set)
        parent_two = self._selection_strat(action_set)
        child_one = copy.deepcopy(parent_one)
        child_two = copy.deepcopy(parent_two)
        child_one.numerosity = 1
        child_two.numerosity = 1
        child_one.experience = 0
        child_two.experience = 0

        parents = ClassifierPair(parent_one, parent_two)
        children = ClassifierPair(child_one, child_two)
        return parents, children

    def _perform_crossover(self, children, parents, situation):
        should_do_crossover = get_rng().rand() < get_hyperparam("chi")
        if should_do_crossover:
            (child_one, child_two) = children
            logging.debug(f"Before crossover {child_one.condition}, "
                          f"{child_two.condition}")
            self._rule_repr.crossover_conditions(child_one.condition,
                                                 child_two.condition,
                                                 self._crossover_strat)
            logging.debug(f"After crossover {child_one.condition}, "
                          f"{child_two.condition}")
            self._update_children_params(children, parents, situation)

    def _update_children_params(self, children, parents, situation):
        (child_one, child_two) = children
        (parent_one, parent_two) = parents

        child_prediction = (parent_one.get_prediction(situation) +
                            parent_two.get_prediction(situation)) / 2
        self._try_update_children_prediction(children, child_prediction)

        child_error = 0.25 * (parent_one.error + parent_two.error) / 2
        child_one.error = child_error
        child_two.error = child_error

        child_fitness = 0.1 * (parent_one.fitness + parent_two.fitness) / 2
        child_one.fitness = child_fitness
        child_two.fitness = child_fitness

    def _try_update_children_prediction(self, children, child_prediction):
        for child in children:
            try:
                child.set_prediction(child_prediction)
            except AttributeError:
                pass

    def _perform_mutation(self, children, situation):
        child_one, child_two = children
        logging.debug(f"Before mutation: {child_one.rule}, {child_two.rule}")
        for child in children:
            self._mutation_strat(child, situation)
        logging.debug(f"After mutation: {child_one.rule}, {child_two.rule}")

    def _update_population(self, children, parents, population):
        should_do_subsumption = get_hyperparam("do_ga_subsumption")
        for child in children:
            if should_do_subsumption:
                was_subsumed = self._try_subsume_with_parents(
                    child, parents, population)
                if not was_subsumed:
                    population.insert(child, operation_label="discovery")
            else:
                population.insert(child, operation_label="discovery")

    def _try_subsume_with_parents(self, child, parents, population):
        for parent in parents:
            if self._subsumption_strat.does_subsume(parent, child):
                assert child.numerosity == 1
                self._try_duplicate_parent_in_population(parent, population)
                return True
        return False

    def _try_duplicate_parent_in_population(self, parent, population):
        try:
            population.duplicate(parent,
                                 num_copies=1,
                                 operation_label="ga_subsumption")
        except MemberNotFoundError:
            logging.debug("GA subsumption failure.")
        else:
            logging.debug("GA subsumption success.")

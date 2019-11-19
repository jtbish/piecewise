import copy
import random
from collections import namedtuple

from .genetic_algorithm import GAOperators, GeneticAlgorithm
from .operator.crossover import TwoPointCrossover
from .operator.mutation import RuleReprMutation
from .operator.selection import RouletteWheelSelection

ClassifierPair = namedtuple("ClassifierPair", ["first", "second"])


class XCSGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, env_action_set, rule_repr, subsumption_strat,
                 hyperparams):
        ga_operators = self._init_ga_operators(env_action_set, rule_repr,
                                               hyperparams)
        super().__init__(env_action_set, subsumption_strat, hyperparams,
                         ga_operators)

    def _init_ga_operators(self, env_action_set, rule_repr, hyperparams):
        selection_strat = RouletteWheelSelection()
        crossover_strat = TwoPointCrossover()
        mutation_strat = RuleReprMutation(rule_repr, env_action_set,
                                          hyperparams)
        return GAOperators(selection_strat, crossover_strat, mutation_strat)

    def __call__(self, operating_set, population, situation, time_step):
        """RUN GA function from 'An Algorithmic Description of
        XCS' (Butz and Wilson, 2002), without initial time check as this is
        factored into the caller.
        """
        action_set = operating_set
        self._update_participant_time_stamps(action_set, time_step)
        parents, children = self._select_parents_and_init_children(action_set)
        self._perform_crossover(children, parents)
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

    def _perform_crossover(self, children, parents):
        do_crossover = random.random() < self._hyperparams["chi"]
        if do_crossover:
            self._crossover_strat(*children)
            self._update_children_params(children, parents)

    def _update_children_params(self, children, parents):
        (child_one, child_two) = children
        (parent_one, parent_two) = parents
        child_one.prediction = \
            (parent_one.prediction + parent_two.prediction)/2
        child_one.error = \
            0.25*(parent_one.error + parent_two.error)/2
        child_one.fitness = \
            0.1*(parent_one.fitness + parent_two.fitness)/2
        child_two.prediction = child_one.prediction
        child_two.error = child_one.error
        child_two.fitness = child_one.fitness

    def _perform_mutation(self, children, situation):
        for child in children:
            self._mutation_strat(child, situation)

    def _update_population(self, children, parents, population):
        try_subsumption = self._subsumption_strat is not None
        for child in children:
            if try_subsumption:
                was_subsumed = self._try_subsume_with_parents(
                    child, parents, population)
                if not was_subsumed:
                    population.insert(child, track_as="discovery")
            else:
                population.insert(child, track_as="discovery")

    def _try_subsume_with_parents(self, child, parents, population):
        for parent in parents:
            if self._subsumption_strat.does_subsume(parent, child):
                assert child.numerosity == 1
                population.duplicate(parent,
                                     num_copies=1,
                                     track_as="ga_subsumption")
                return True
        return False

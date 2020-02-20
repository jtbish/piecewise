from piecewise.lcs.rng import get_rng


def roulette_wheel_selection(operating_set):
    """SELECT OFFSPRING function from 'An Algorithmic Description of XCS'
    (Butz and Wilson, 2002)."""
    fitness_sum = sum([classifier.fitness for classifier in operating_set])
    choice_point = get_rng().rand() * fitness_sum

    fitness_sum = 0
    for classifier in operating_set:
        fitness_sum += classifier.fitness
        if fitness_sum > choice_point:
            return classifier

from gnas.search_space.individual import Individual, MultipleBlockIndividual
from gnas.genetic_algorithm.cross_over import uniform_crossover, select_crossover


def _individual_uniform_crossover(individual_a: Individual, individual_b: Individual) -> Individual:
    res = [uniform_crossover(a, b) for a, b in zip(individual_a.iv, individual_b.iv)]
    return Individual(res, individual_a.mi, individual_a.ss)


def individual_uniform_crossover(individual_a, individual_b):
    if isinstance(individual_a, Individual):
        return _individual_uniform_crossover(individual_a, individual_b)
    else:
        return MultipleBlockIndividual([_individual_uniform_crossover(inv_a, inv_b) for inv_a, inv_b in
                                        zip(individual_a.individual_list, individual_b.individual_list)])

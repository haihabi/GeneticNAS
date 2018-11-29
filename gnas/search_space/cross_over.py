import numpy as np
from gnas.search_space.individual import Individual
from gnas.genetic_algorithm.cross_over import uniform_crossover, uniform_crossover_sum_one


def individual_uniform_crossover(individual_a: Individual, individual_b: Individual) -> Individual:
    res = [uniform_crossover(a, b) for a, b in zip(individual_a.iv, individual_b.iv)]

    return Individual(res, individual_a.mi, individual_a.ss)

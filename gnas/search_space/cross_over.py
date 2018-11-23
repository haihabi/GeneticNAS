import numpy as np
from gnas.search_space.individual import Individual
from gnas.genetic_algorithm.cross_over import uniform_crossover, uniform_crossover_sum_one


def individual_uniform_crossover(individual_a: Individual, individual_b: Individual) -> Individual:
    operation_vector = uniform_crossover(np.concatenate(individual_a.operation_vector),
                                         np.concatenate(individual_b.operation_vector))
    connection_vector = []
    for ca, cb in zip(individual_a.connection_vector, individual_b.connection_vector):
        connection_vector.append(uniform_crossover_sum_one(ca, cb))
    return Individual(connection_vector, np.hsplit(operation_vector, individual_a.n_iterations), individual_a.oc)

import numpy as np
from gnas.search_space.individual import Individual
from gnas.genetic_algorithm.mutation import flip_bit, flip_bit_sum_one


def individual_flip_mutation(individual_a: Individual, p) -> Individual:
    operation_vector = flip_bit(np.concatenate(individual_a.operation_vector), p)
    connection_vector = []
    for ca in individual_a.connection_vector:
        connection_vector.append(flip_bit_sum_one(ca, p))
    return Individual(connection_vector, np.hsplit(operation_vector, individual_a.n_iterations), individual_a.oc)

import numpy as np
from gnas.search_space.individual import Individual


def flip_max_value(current_value, max_value, p):
    flip = 1 - np.round(0.5 - p + np.random.rand(current_value.shape[0])).astype('int')
    new_dna = current_value + flip
    new_dna[new_dna > max_value] = 0
    return new_dna


def individual_flip_mutation(individual_a: Individual, p) -> Individual:
    max_values = individual_a.ss.get_max_values_vector()
    new_iv = []
    for m, iv in zip(max_values, individual_a.iv):
        new_iv.append(flip_max_value(iv, m, p))
    return individual_a.update_individual(new_iv)

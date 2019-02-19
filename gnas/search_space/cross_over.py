import numpy as np
from gnas.search_space.individual import Individual, MultipleBlockIndividual

def _individual_uniform_crossover(individual_a: Individual, individual_b: Individual):
    n = individual_a.get_length()
    selection = np.random.randint(0, 2, n)
    i = 0
    iv_a = []
    iv_b = []
    for a, b in zip(individual_a.iv, individual_b.iv):
        current_selection = selection[i:i + len(a)]
        iv_a.append(a * current_selection + b * (1 - current_selection))
        iv_b.append(a * (1 - current_selection) + b * current_selection)
        i += len(a)
    return Individual(iv_a, individual_a.mi, individual_a.ss, index=individual_a.index), \
           Individual(iv_b, individual_b.mi, individual_b.ss, index=individual_b.index)


def _individual_block_crossover(individual_a: Individual, individual_b: Individual):
    n = individual_a.get_n_op()
    selection = np.random.randint(0, 2, n)
    iv_a = []
    iv_b = []
    for i, (a, b) in enumerate(zip(individual_a.iv, individual_b.iv)):
        # current_selection = selection[i]
        iv_a.append(a * selection[i] + b * (1 - selection[i]))
        iv_b.append(a * (1 - selection[i]) + b * selection[i])

    return Individual(iv_a, individual_a.mi, individual_a.ss, index=individual_a.index), \
           Individual(iv_b, individual_b.mi, individual_b.ss, index=individual_b.index)


def individual_uniform_crossover(individual_a, individual_b, p_c):
    if np.random.rand(1) < p_c:
        if isinstance(individual_a, Individual):
            return _individual_uniform_crossover(individual_a, individual_b)
        else:
            pairs = [_individual_uniform_crossover(inv_a, inv_b) for inv_a, inv_b in
                     zip(individual_a.individual_list, individual_b.individual_list)]
            return MultipleBlockIndividual([p[0] for p in pairs]), MultipleBlockIndividual([p[1] for p in pairs])
    else:
        return individual_a, individual_b


def individual_block_crossover(individual_a, individual_b, p_c):
    if np.random.rand(1) < p_c:
        if isinstance(individual_a, Individual):
            return _individual_block_crossover(individual_a, individual_b)
        else:
            pairs = [_individual_block_crossover(inv_a, inv_b) for inv_a, inv_b in
                     zip(individual_a.individual_list, individual_b.individual_list)]
            return MultipleBlockIndividual([p[0] for p in pairs]), MultipleBlockIndividual([p[1] for p in pairs])
    else:
        return individual_a, individual_b

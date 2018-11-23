import numpy as np


def uniform_crossover(array_a: np.ndarray, array_b: np.ndarray):
    if len(array_a.shape) != 1: raise Exception('')
    if len(array_b.shape) != 1: raise Exception('')
    if array_b.shape[0] != array_a.shape[0]: raise Exception('')
    selection = np.random.randint(0, 1, array_a.shape[0])
    return array_a * selection + array_b * (1 - selection)


def uniform_crossover_sum_one(array_a: np.ndarray, array_b: np.ndarray):
    output_array = uniform_crossover(array_a, array_b)
    if np.all(array_a + array_b < 2):
        select_from = np.where(array_a + array_b == 1)[0]
        output_array[np.random.choice(select_from)] = 1
    return output_array

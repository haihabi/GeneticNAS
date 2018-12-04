import numpy as np


def cosine_annealing(iteration, cycles, delay, end):
    if iteration < delay:
        return 1.0
    elif iteration >= end:
        return 0.0
    else:
        norm_iteration = iteration - delay
        cosine_size = end - delay
        mod_cycle = int(cosine_size / cycles)
        return np.cos(np.pi * (norm_iteration % mod_cycle) / (2 * cosine_size))

import numpy as np


def flip_bit(input_array: np.ndarray, p: float):
    flip = np.round(0.5 - p + np.random.rand(input_array.shape[0])).astype('int')
    output_array = input_array.copy()
    output_array[flip] = 1 - input_array[flip]
    return output_array


def flip_bit_sum_one(input_array: np.ndarray, p: float):
    output_array = flip_bit(input_array, p)
    if np.sum(output_array) == 0:
        output_array[np.random.randint(0, input_array.shape[0])] = 1
    return output_array

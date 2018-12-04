import unittest
import numpy as np
from matplotlib import pyplot as plt
from gnas.genetic_algorithm.annealing_functions import cosine_annealing


class TestAnnealing(unittest.TestCase):
    def test_cosine(self):
        debug_plot = False
        res = np.asarray([cosine_annealing(i, 1, 15, 120) for i in range(150)])
        if debug_plot:
            plt.plot(res)
            plt.show()

    if __name__ == '__main__':
        unittest.main()

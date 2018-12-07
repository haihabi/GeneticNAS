import numpy as np
import unittest
from random import choices
from gnas.genetic_algorithm.genetic import GeneticAlgorithms
from gnas.genetic_algorithm.mutation import flip_bit
from gnas.genetic_algorithm.cross_over import uniform_crossover
import gnas


class TestGenetic(unittest.TestCase):

    def test_basic_problem(self):
        n = 250
        population_size = 20

        def population_initializer(p_size):
            return np.round(np.random.rand(p_size, 4))

        def mutation_function(x):
            return flip_bit(x, 1 / n)

        def cross_over_function(x0, x1):
            return uniform_crossover(x0, x1)

        def selection_function(loss):
            couples = choices(population=list(range(population_size)), weights=loss / np.sum(loss),
                              k=2 * population_size)
            return np.reshape(np.asarray(couples), [-1, 2])

        def objective_function(individual):
            return np.sum([np.power(2, i) * v for i, v in enumerate(individual)])

        # mutation_function, cross_over_function, selection_function
        ga = GeneticAlgorithms(population_initializer, mutation_function, cross_over_function, selection_function)
        population = ga.population
        for i in range(10):
            self.assertTrue(np.sum(np.isnan(ga.population_fitness)) == (population_size - (i % population_size)))
            if i != 0 and i % population_size == 0:
                self.assertTrue(np.sum(np.abs(ga.population - population)) != 0)
                population = ga.population
            else:
                self.assertTrue(np.sum(np.abs(ga.population - population)) == 0)
            ga.update_current_individual_fitness(objective_function(ga.get_current_individual()))

    def test_search_space(self):
        ss = gnas.get_enas_rnn_search_space(12)
        ga = gnas.genetic_algorithm_searcher(ss, population_size=20)
        for i in range(10):
            ga.update_current_individual_fitness(0 + np.random.rand(1))


if __name__ == '__main__':
    unittest.main()

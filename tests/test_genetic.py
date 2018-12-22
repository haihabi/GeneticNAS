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
        n_bits = 20

        def population_initializer(p_size):
            return np.round(np.random.rand(p_size, n_bits))

        def mutation_function(x):
            return flip_bit(x, 1 / n_bits)

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

        for e in range(310):
            population = ga.population
            for i, inv in enumerate(ga.get_current_generation()):
                self.assertTrue(np.sum(np.isnan(ga.population_fitness)) == (population_size - (i % population_size)))
                if i != 0 and i % population_size == 0:
                    self.assertTrue(np.sum(np.abs(ga.population - population)) != 0)
                    population = ga.population
                else:
                    self.assertTrue(np.sum(np.abs(ga.population - population)) == 0)
                ga.update_current_individual_fitness(inv, objective_function(inv))
            ga.update_population()
        print("a")

    def test_search_space(self):
        ss = gnas.get_enas_cnn_search_space(5)
        ga = gnas.genetic_algorithm_searcher(ss, population_size=200, generation_size=20)
        for i in range(300):
            for i, ind in enumerate(ga.get_current_generation()):
                ga.sample_child(0.2)
                ga.update_current_individual_fitness(ind, 0 + np.random.rand(1))
            ga.update_population()
            print(len(ga.max_dict))
            self.assertTrue(len(ga.max_dict) <= 200)
            self.assertTrue(len(ga.generation))


if __name__ == '__main__':
    unittest.main()

import numpy as np
import pickle
import os
from random import choices
from gnas.search_space.search_space import SearchSpace
from gnas.search_space.cross_over import individual_uniform_crossover
from gnas.search_space.mutation import individual_flip_mutation
from gnas.genetic_algorithm.ga_results import GenetricResult


def genetic_algorithm_searcher(search_space: SearchSpace, population_size=10, elitism=True):
    def population_initializer(p_size):
        return search_space.generate_population(p_size)

    def mutation_function(x):
        return individual_flip_mutation(x, 1 / 50)

    def cross_over_function(x0, x1):
        return individual_uniform_crossover(x0, x1)

    def selection_function(p):
        couples = choices(population=list(range(population_size)), weights=p,
                          k=2 * population_size)
        return np.reshape(np.asarray(couples), [-1, 2])

    return GeneticAlgorithms(population_initializer, mutation_function, cross_over_function, selection_function,
                             min_objective=True, population_size=population_size,
                             elitism=elitism)


class GeneticAlgorithms(object):
    def __init__(self, population_initializer, mutation_function, cross_over_function, selection_function,
                 population_size=20, min_objective=False,
                 elitism=False):
        self.population_initializer = population_initializer
        self.mutation_function = mutation_function
        self.cross_over_function = cross_over_function
        self.selection_function = selection_function
        self.elitism = elitism
        self.population_size = population_size
        self.i = 0
        self.ga_result = GenetricResult()

        self.population = None
        self.population_fitness = None
        self.min_objective = min_objective
        self._init_population()

    def _init_population(self):
        self.population = self.population_initializer(self.population_size)
        self.population_fitness = np.nan * np.ones(self.population_size)

    def update_population(self):
        self.ga_result.add_result(self.population_fitness, self.population)
        print(self.population_fitness)
        f_mean = np.mean(self.population_fitness)
        f_var = np.var(self.population_fitness)
        f_max = np.max(self.population_fitness)
        f_min = np.min(self.population_fitness)
        best_individual = self.population[np.nanargmax(self.population_fitness)]
        p = self.population_fitness / np.nansum(self.population_fitness)
        p[np.isnan(p)] = 0
        if self.min_objective:
            p = 1 - p
            best_individual = self.population[np.nanargmin(self.population_fitness)]

        couples = self.selection_function(p)  # selection
        child = [self.cross_over_function(self.population[c[0]], self.population[c[1]]) for c in couples]  # cross-over
        self.population = np.asarray([self.mutation_function(c) for c in child])  # mutation
        self.population_fitness = np.nan * np.ones(self.population_size)  # clear fitness results
        if self.elitism:
            best_index = np.random.random_integers(0, self.population_size - 1)
            self.population[best_index] = best_individual
        # update generation index and individual index
        self.i = 0
        print(
            "Update population | mean fitness: {:5.2f} | var fitness {:5.2f} || max fitness: {:5.2f} | min fitness {:5.2f} |".format(
                f_mean, f_var, f_max, f_min))
        return f_mean, f_var, f_max, f_min

    def get_current_individual(self):
        current_individuals = self.population[self.i % self.population_size]
        self.i += 1
        return current_individuals

    def update_current_individual_fitness(self, individual_fitness):
        self.population_fitness[(self.i - 1) % self.population_size] = individual_fitness

    def sample_child(self, p):
        if p > np.random.rand(1):
            return self.population_initializer(1)[0]
        else:
            couple = np.random.randint(0, self.population_size, 2)
            child = self.cross_over_function(self.population[couple[0]], self.population[couple[1]])
            return self.mutation_function(child)

    def get_result(self):
        return self.ga_result.fitness_list, self.ga_result.population_list

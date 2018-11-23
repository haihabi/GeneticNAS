import numpy as np


class GeneticAlgorithms(object):
    def __init__(self, population_initializer, mutation_function, cross_over_function, selection_function,
                 population_size=20, n_generation=100,
                 elitism=False):
        self.population_initializer = population_initializer
        self.mutation_function = mutation_function
        self.cross_over_function = cross_over_function
        self.selection_function = selection_function
        self.elitism = elitism
        self.population_size = population_size
        self.n_generation = n_generation
        self.i = 0
        self.g_index = 0
        self.population = None
        self.population_fitness = None
        self._init_population()

    def __iter__(self):
        return self

    def __next__(self):
        if self.g_index < self.n_generation:
            if (self.i % self.population_size) == 0 and self.i != 0:
                self._update_population()
            current_individuals = self.population[self.i % self.population_size]
            self.i += 1
            return current_individuals
        else:  # finished all generations
            raise StopIteration

    def _init_population(self):
        self.population = self.population_initializer()
        self.population_fitness = np.nan * np.ones(20)

    def _update_population(self):
        print("Update population")
        best_individual = self.population[np.argmax(self.population_fitness)]
        couples = self.selection_function(self.population_fitness)  # selection
        child = [self.cross_over_function(self.population[c[0]], self.population[c[1]]) for c in couples]  # cross-over
        self.population = np.asarray([self.mutation_function(c) for c in child])  # mutation
        self.population_fitness = np.nan * np.ones(20)  # clear fitness results
        if self.elitism:
            best_index = np.random.random_integers(0, self.population_size)
            self.population[best_index] = best_individual
        # update generation index and individual index
        self.i = 0
        self.g_index += 1

    def update_current_individual_fitness(self, individual_fitness):
        self.population_fitness[(self.i - 1) % self.population_size] = individual_fitness

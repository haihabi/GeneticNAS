import numpy as np
from random import choices
from gnas.search_space.search_space import SearchSpace
from gnas.search_space.cross_over import individual_uniform_crossover, individual_block_crossover
from gnas.search_space.mutation import individual_flip_mutation
from gnas.genetic_algorithm.ga_results import GenetricResult
from gnas.genetic_algorithm.population_dict import PopulationDict


def genetic_algorithm_searcher(search_space: SearchSpace, generation_size=20, population_size=300, keep_size=0,
                               min_objective=True, mutation_p=None, p_cross_over=None, cross_over_type='Bit'):
    if mutation_p is None: mutation_p = 1 / search_space.n_elements
    if p_cross_over is None: p_cross_over = 1
    print('p mutation:' + str(mutation_p), 1 / search_space.n_elements)

    def population_initializer(p_size):
        return search_space.generate_population(p_size)

    def mutation_function(x):
        return individual_flip_mutation(x, mutation_p)

    if cross_over_type == 'Bit':
        print("Bit base cross over")

        def cross_over_function(x0, x1):
            return individual_uniform_crossover(x0, x1, p_cross_over)
    elif cross_over_type == 'Block':
        print("Block base cross over")

        def cross_over_function(x0, x1):
            return individual_block_crossover(x0, x1, p_cross_over)
    else:
        raise Exception('')

    def selection_function(p):
        couples = choices(population=list(range(len(p))), weights=p,
                          k=generation_size)
        return np.reshape(np.asarray(couples), [-1, 2])

    return GeneticAlgorithms(population_initializer, mutation_function, cross_over_function, selection_function,
                             min_objective=min_objective, generation_size=generation_size,
                             population_size=population_size, keep_size=keep_size)


class GeneticAlgorithms(object):
    def __init__(self, population_initializer, mutation_function, cross_over_function, selection_function,
                 population_size=300, generation_size=20, keep_size=20, min_objective=False):
        ####################################################################
        # Functions
        ####################################################################
        self.population_initializer = population_initializer
        self.mutation_function = mutation_function
        self.cross_over_function = cross_over_function
        self.selection_function = selection_function
        ####################################################################
        # parameters
        ####################################################################
        self.population_size = population_size
        self.generation_size = generation_size
        self.keep_size = keep_size
        self.min_objective = min_objective
        ####################################################################
        # status
        ####################################################################
        self.max_dict = PopulationDict()
        self.ga_result = GenetricResult()
        self.current_dict = dict()

        self.generation = self._create_random_generation()

        self.i = 0
        self.best_individual = None

    def _create_random_generation(self):
        return self.population_initializer(self.generation_size)

    def _create_new_generation(self, population, population_fitness):
        p = population_fitness / np.nansum(population_fitness)
        if self.min_objective: p = 1 - p
        couples = self.selection_function(p)  # selection
        child = [cc for c in couples for cc in
                 self.cross_over_function(population[c[0]], population[c[1]])]  # cross-over
        new_generation = np.asarray([self.mutation_function(c) for c in child])  # mutation

        p_array = np.asarray([p.code for p in new_generation])
        b = np.ascontiguousarray(p_array).view(np.dtype((np.void, p_array.dtype.itemsize * p_array.shape[1])))
        _, idx = np.unique(b, return_index=True)
        if len(idx) == self.generation_size:
            generation = new_generation
        else:
            n = self.generation_size - len(idx)
            p_new = self.population_initializer(n)
            generation = np.asarray([*[new_generation[i] for i in idx], *p_new])
        return generation

    def update_population(self):
        self.i += 1

        generation_fitness = np.asarray(list(self.current_dict.values()))
        generation = list(self.current_dict.keys())
        self.ga_result.add_generation_result(generation_fitness, generation)

        f_mean = np.mean(generation_fitness)
        f_var = np.var(generation_fitness)
        f_max = np.max(generation_fitness)
        f_min = np.min(generation_fitness)
        total_dict = self.max_dict.copy()
        total_dict.update(self.current_dict)
        # last_dict = None
        # if self.keep_size > 0:
        #     last_dict = total_dict.filter_last_n(self.keep_size)
        # if self.population_size - self.keep_size > 0:

        best_max_dict = total_dict.filter_top_n(self.population_size,min_max=not self.min_objective)
        n_diff = self.max_dict.get_n_diff(best_max_dict)
        self.max_dict = best_max_dict
        #
        #     if self.keep_size > 0:
        #         best_max_dict = best_max_dict.merge(last_dict)
        # else:
        #     best_max_dict = last_dict




        self.current_dict = dict()
        population_fitness = np.asarray(list(self.max_dict.values())).flatten()
        population = np.asarray(list(self.max_dict.keys())).flatten()
        self.best_individual = population[np.argmax(population_fitness)]
        fp_mean = np.mean(population_fitness)
        fp_var = np.var(population_fitness)
        fp_max = np.max(population_fitness)
        fp_min = np.min(population_fitness)
        self.ga_result.add_population_result(population_fitness, population)
        self.generation = self._create_new_generation(population, population_fitness)


        print(
            "Update generation | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |population size {:d}|".format(
                f_mean, f_var, f_max, f_min, len(population)))
        print(
            "population results | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |".format(
                fp_mean, fp_var, fp_max, fp_min))
        return f_mean, f_var, f_max, f_min, n_diff

    def get_current_generation(self):
        return self.generation

    def update_current_individual_fitness(self, individual, individual_fitness):
        self.current_dict.update({individual: individual_fitness})

    def sample_child(self):
        if len(list(self.max_dict.keys())) == 0: # if not population exist generate random indivaul
            return self.population_initializer(1)[0]
        else:
            couples = choices(list(self.max_dict.keys()), k=2) # random select two indivuals from population
            child = self.cross_over_function(couples[0], couples[1]) # prefome cross over
            return self.mutation_function(child[0]) # select the first then mutation

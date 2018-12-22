class GenetricResult(object):
    def __init__(self):
        self.population_list = []
        self.fitness_list = []
        self.fitness_full_list = []
        self.population_full_list = []

    def add_generation_result(self, fitness, population):
        self.fitness_list.append(fitness)
        self.population_list.append(population)

    def add_population_result(self, fitness, population):
        self.fitness_full_list.append(fitness)
        self.population_full_list.append(population)

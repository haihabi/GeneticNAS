class GenetricResult(object):
    def __init__(self):
        self.population_list = []
        self.fitness_list = []

    def add_result(self, fitness, population):
        self.fitness_list.append(fitness)
        self.population_list.append(population)

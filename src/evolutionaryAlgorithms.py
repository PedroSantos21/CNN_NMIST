class GeneticAlgorithm():

    def __init__(self,
        parameter_space,
        population_size,
        elitism         = 0.3,
        crossover_rate  = 0.1,
        mutation_rate   = 0.25,
    ):
        self.elitism = elitism
        self.crossover_rate = crossover_rate
        self.mutation_rate  = mutation_rate
        self.population_size = population_size

    
    def createPopulation(self):
        break




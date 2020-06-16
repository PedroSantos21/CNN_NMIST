import numpy as np
from random import random, randint, uniform, choice, choices, sample
from tqdm import tqdm
from time import time
import sys


class GeneticAlgorithm:

    def __init__(self,
        parameters,
        fitnessFunction,
        population_size,
        generations,
        elitism         = 0.1,
        crossover_rate  = 0.8,
        crossoverPoint  = None,
        mutation_rate   = 0.25,
        random_selection_rate = 0.01
    ):
        self.elitism = elitism
        self.generations = generations
        self.populationSize = population_size
        self.crossoverRate = crossover_rate
        self.mutationRate  = mutation_rate
        self.randoSelectionRate = random_selection_rate
        
        self.parametersRange = list(parameters.values())
        self.fitnessFunction = fitnessFunction
        
        # CROSSOVER
        self.crossoverPoint = int(len(parameters)/2) if crossoverPoint == None else crossoverPoint

        # AUX
        self.precision = 7

        # self.population = self.createPopulation()
    

    # POPULATIONAL SECTION
    def createIndividual(self):
        '''
        Individual is represented as a possible solution 
        to the problem.

        In this case a solution is an array with values of
        the selected hyperparameters.

        A probability distribution (random, gaussian, uniform) is the best way to generate
        values inside a range of possible values
        '''
        return [round(uniform(*parameter), 3) if type(parameter) == tuple else choice(parameter)
            for parameter in self.parametersRange]
    
    def individualFormat(self, individual):
        return tuple(individual)
    
    def createPopulation(self):
        '''
        Create an initial random population according with the
        parameters of the problem and its valid values
        '''
        print("Creating initial random population...")
        population = []
        while len(population) < self.populationSize:
            ind = self.createIndividual()
            print(ind)
            population.append(ind)
        return population

    #   FITNESS SECTION
    def fitness(self, individual):
        '''
        function fitness = evaluate_individual
        '''
        ind = self.individualFormat(individual)
        return self.fitnessFunction(ind)


    def sortByFitness(self, population):
        scores = [self.fitness(individual) for individual in tqdm(population, desc="Measuring Population Fitness", file=sys.stdout)]
        return [x for _, x in sorted(zip(scores, population), key=lambda p: p[0], reverse=True)]

    def populationFitness(self, population):
        return [self.fitness(individual) for individual in tqdm(population, desc="Measuring Population Fitness", file=sys.stdout)]    
    
    def orderPopulation(self, scores, population):
        self.scores, self.population = [list(t) for t in zip(*sorted(zip(scores, population)))]   


    def grade(self, list_fit=None):
        '''
        Find minimum fitness for a population.
        '''
        if not list_fit:
            list_fit = self.scores
        try:
            return np.nanmin([fit for fit in self.scores])
        except:
            return np.nan
    
    # REPRODUCTION SECTION
    def crossover(self, individual1, individual2):

        child1 = individual1.copy()
        child2 = individual2.copy()

        if np.random.uniform(0,1) < self.crossoverRate:
            child1 = individual1[:self.crossoverPoint] + individual2[self.crossoverPoint:]
            child2 = individual2[:self.crossoverPoint] + individual1[self.crossoverPoint:]

        return child1, child2


    # MUTATION SECTION
    def mutation(self, individual):
        if np.random.uniform(0,1) < self.mutationRate:
            locus = randint(0, len(individual)-1)
            parameter = self.parametersRange[locus]
            individual[locus] = uniform(*parameter) if type(parameter) == tuple else choice(parameter)



    # GENERATIONAL SECTION
    def evolve(self):

        # ELITISMO
        elitismSize = int(self.populationSize*self.elitism)
        # orderedPop = self.sortByFitness(population)
        newGeneration = [ind for ind in tqdm(self.population[:elitismSize], desc="Applying Elitism", file=sys.stdout)]

        while len(newGeneration) < self.populationSize:
            
            # RANDOM SELECTION (DIVERSITY)
            for individual in tqdm(self.population[elitismSize:], desc="Random Selection", file=sys.stdout):
                if np.random.uniform(0,1) < self.randoSelectionRate:
                    newGeneration.append(individual)
        
            # RANDOM MUTATION (DIVERSITY)
            for individual in tqdm(self.population[elitismSize:], desc="Random Mutation", file=sys.stdout):
                self.mutation(individual)
                newGeneration.append(individual)

            # CROSSOVER
            ind1, ind2 = sample(self.population, 2)

            child1, child2 = self.crossover(ind1, ind2)

            if np.random.uniform(0,1) < self.mutationRate:
                randomSelection = choice([child1, child2])
                self.mutation(randomSelection)
                newGeneration.append(randomSelection)            
            newGeneration.append(child1)
            newGeneration.append(child2)

        # EVALUATE POPULATION
        generationScores = self.populationFitness(newGeneration)
        generationbestFitness = self.grade(generationScores) 

        print("Best fitness of this generation:", generationbestFitness)

        self.orderPopulation(generationScores, newGeneration)
        self.bestFitness = generationbestFitness

        

    def populationInfo(self, population):
        pass
        

    def run(self):
        
        counter = 0
        # CREATE INITIAL RANDOM POPULATION
        self.population = self.createPopulation()

        # EVALUATE INITIAL POPULATION
        self.scores = self.populationFitness(self.population)
        self.bestFitness = self.grade() 
        print("Initial best fitness:", self.bestFitness)
        
        # ORGANIZING POPULATION BY FITNESS
        self.orderPopulation(self.scores, self.population)
        
        while counter < self.generations:
            print(f"\n  Running iteration {(counter+1)}/{self.generations}")

            self.evolve()

            counter += 1
        
        return self.bestFitness
        

    

                




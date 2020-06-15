import numpy as np
from random import random, randint, uniform, choice, choices
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
        crossover_rate  = 0.2,
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

        self.population = self.createPopulation()
    

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
        population = []
        while len(population) < self.populationSize:
            ind = self.createIndividual()
            print(ind)
            population.append(ind)
        return population

    #   FITNESS SECTION
    def fitness(self, individual):
        ind = self.individualFormat(individual)
        return self.fitnessFunction(ind)

    def sortByFitness(self, population):
        scores = [self.fitness(individual) for individual in population]
        return [x for _, x in sorted(zip(scores, population), key=lambda p: p[0], reverse=True)]

    
    # REPRODUCTION SECTION
    def crossover(self, individual1, individual2):

        child1 = individual1.copy()
        child2 = individual2.copy()

        if np.random.uniform(0,1) < self.crossoverRate:

            child1[self.crossoverPoint:] = child1[self.crossoverPoint:] + child2[self.crossoverPoint:]
            child2[self.crossoverPoint:] = child1[self.crossoverPoint:] - child2[self.crossoverPoint:]
            child1[self.crossoverPoint:] = child1[self.crossoverPoint:] - child2[self.crossoverPoint:]

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
        orderedPop = self.sortByFitness(self.population)
        newGeneration = [ind for ind in tqdm(orderedPop[:elitismSize], desc=" Applying Elitism", file=sys.stdout)]

        while len(newGeneration) < self.populationSize:
            
            # RANDOM SELECTION (DIVERSITY)
            for individual in tqdm(orderedPop[elitismSize:], desc="    Random Selection", file=sys.stdout):
                if np.random.uniform(0,1) < self.randoSelectionRate:
                    newGeneration.append(individual)
        
            # RANDOM MUTATION (DIVERSITY)
            for individual in tqdm(orderedPop[elitismSize:], desc="    Random Mutation", file=sys.stdout):
                self.mutation(individual)
                newGeneration.append(individual)

            # CROSSOVER
            ind1, ind2 = choices(self.population,k=2)

            child1, child2 = self.crossover(ind1, ind2)

            if np.random.uniform(0,1) < self.mutationRate:
                randomSelection = choice([child1, child2])
                self.mutation(randomSelection)
                newGeneration.append(randomSelection)            
            newGeneration.append(child1)
            newGeneration.append(child2)

        self.populationInfo(newGeneration)
        
        return newGeneration

    def populationInfo(self, population):
        orderedPop = self.sortByFitness(population)
        print("\n New population:", orderedPop)

        bestFitness = self.fitness(orderedPop[0])
        print("    Best fitness of this generation:", bestFitness)
    

    def run(self):
        self.population = self.evolve()
        self.populationInfo(self.population)
        
        return self.sortByFitness(self.population)[0]
        

    

                




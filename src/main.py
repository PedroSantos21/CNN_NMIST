import sys
from utils import *
from classifier import Classifier
from evolutionaryAlgorithms import GeneticAlgorithm as GA

# sys.path.insert(0, '..')
# sys.path
# from EvoLSTM.Lib.ga import GA
from datetime import datetime
from os import makedirs
import traceback
import shutil
import sys
import numpy as np
from getpass import getpass
import smtplib
import logging
from time import time



def run(**kwargs):
    algorithm = kwargs.get('algorithm')
    dataset = kwargs.get('dataset')

    if kwargs.get('algorithm') == 'GA':
        fitness = kwargs.get('fitness')
        parameters = kwargs.get('parameters')
        popSize = kwargs.get('population_size')
        generations = kwargs.get('generations')
        history = {}
    
        

        # evolver = GA(fitness, parameters, popSize, generations, history)
        evolver = GA(parameters=parameters, fitnessFunction=fitness, population_size=popSize, generations=generations)
    else:
        pass
            # evolver = algorithms[algorithm](fitness, parameters, popSize, num_it, normParam, history)

    # create results dir
    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    path = f"{algorithm}/results_NMIST_{timestamp}"
    makedirs(path)
        
    with open(f"{path}/{algorithm}_results.txt", "w+") as f:

        results = []
        loss = []
        for i in range(generations):
            print(f"\nRunning execution {(i+1)}/{generations}")
            # Run Evolver
            best = evolver.run()
            print('BEST GENE', best['gene'])
            # Calculate loss
            gen_loss = evolver.fitness(best['gene'], test=True, batch_size=batch_size, epochs=epochs)
            print('gen_loss', gen_loss)
            loss.append(gen_loss)
            # Store results
            results.append({
                'best': best,
                'gen_loss': gen_loss,
                'hist': hist, 
                'pop': evolver.pop, 
                'fit': evolver.fit, 
                'history': evolver.history
            })
        return loss, results



def main():
    # GLOBAL GA PARAMETERS  
    GENERATIONS             = 2
    POPULATION_SIZE         = 4
    MUTATION_RATE           = 0.4
    CROSSOVER_RATE          = 0.8

    # GLOBAL CNN PARAMETERS
    EPOCHS                  = 10
    BATCH_SIZE              = 256


        # Hiperparametros:
    # - Learning rate
    # - Funcao de otimizacao
    # - Tamanho camada 1
    # - Tamanho camada 2
    # - Tamanho camada 3

    # Fitness:
    # - loss
    # - accuracy

    # Intervalo do tamanho da camada [2, 1024]

    parameters = {
        'learningRate': (0.001, 0.1),
        'optimizer': ['Adam', 'Sgd', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax'],
        'cnnSize_1': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        'cnnSize_2': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        'cnnSize_3': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    }

    normParam = {
        'cnnSize_1': convertPow2,
        'cnnSize_2': convertPow2,
        'cnnSize_3': convertPow2
    }

    train_data, train_labels_cat, \
    test_data, test_labels_cat, \
    validation_data, validation_labels_cat = load_dataset_with_validation()

    # Instantiate CNN Clssifier with the MNIST dataset
    cnn = Classifier(
        train_digits        =train_data,
        train_labels        =train_labels_cat,
        validation_digits   =validation_data,
        validation_labels   =validation_labels_cat,
        test_digits         =test_data,
        test_labels         =test_labels_cat,
        verbose             =1)

    def fitness(individual, test=False):
        cnn.clear()

        cnn.configureArchitecture(dict(zip(parameters.keys(), individual)) if not isinstance(individual, dict) else individual)

        cnn.fit(batchSize=BATCH_SIZE, nbEpochs=EPOCHS)

        results = cnn.evaluate(test)
        return results['loss']

    

    run(algorithm='GA', dataset='NMIST',fitness=fitness, parameters=parameters, population_size=POPULATION_SIZE, generations=GENERATIONS)







if __name__ == '__main__':
    main()

    








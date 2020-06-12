import sys
from utils import *
from classifier import Classifier

sys.path.insert(0, '..')
sys.path
from EvoLSTM.Lib.ga import GA
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

        evolver = GA(fitness, parameters, popSize, generations, history)
    else:
        pass
            # evolver = algorithms[algorithm](fitness, parameters, popSize, num_it, normParam, history)

    try:
        # create results dir
        timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        path = f"{algorithm}/results_{dataset}_{timestamp}"
        makedirs(path)
        imgPath = path + "/plots"
        makedirs(imgPath)

        # get TF logger
        log = logging.getLogger('tensorflow')
        log.setLevel(logging.DEBUG)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # create file handler which logs even debug messages
        fh = logging.FileHandler('tensorflow.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)

        loss = []
                
        with open(f"{path}/{algorithm}_results.txt", "w+") as f:
            results = []
            for i in range(generations):
                print(f"\nRunning generation {(i+1)}/{generations}")
                # Run Evolver
                best, hist = evolver.run()
                # Normalize best gene
                if algorithm != 'GA':
                    best['gene'] = best['gene'].norm_eval()
                print('BEST GENE', best['gene'])
                # Calculate loss and accuracy
                gen_loss = fitness(best['gene'], test=True)
                

                print('gen_loss', gen_loss)

                loss.append(gen_loss)
                
                # Normalize population
                if algorithm != 'GA':
                    pop = [individual.norm_eval() for individual in gp.pop]
                # Plot results
                try:
                    evolver.plot(hist, "save", f"{imgPath}/plot{i+1}_en.pdf")
                except:
                    # TODO Exception handler
                    pass
                # Store results
                results.append({
                    'best': best,
                    'gen_loss': gen_loss,
                    'hist': hist, 
                    'pop': evolver.pop, 
                    'fit': evolver.fit, 
                    'history': evolver.history
                })
                f.write(str(results))

        # Calculate stats
        mean = np.nanmean(loss)
        std = np.nanstd(loss)

        # Store stats         
        with open(f"{path}/report.txt", "w+") as f:
            f.write(f"{algorithm} - Mean: {mean} | Std: {std}\n")
            
        print("Success")
        error = False

    except:
        '''
            Print current execution data
        '''
        error = traceback.format_exc()
        print(locals()) # local variables
        print(error, file=sys.stderr) # exception trace
        print('{algorithm} Current Population', file=sys.stderr)
        print([individual.norm_eval() for individual in evolver.pop], file=sys.stderr)
        print('\n{algorithm} Current Results', file=sys.stderr)
        print(*results[:-1], sep="\n", file=sys.stderr)

    finally:
        print("FIM")
        sys.exit()
        # # Store results
        # with open(f'datasets/{dataset}_history', 'w+') as f:
        #     f.write(str(history))

        # Ask to stop and delete results if error
        # if error:
        #     op = input("Delete execution dir? (y/n) ")
        #     if op != 'y' and op != 'n':
        #         print("Invalid option")
        #         op = input("Delete execution dir? (y/n) ")
        #     if op == 'y':
        #         shutil.rmtree(path)
        #     op = input("Continue? (y/n) ")
        #     if op != 'y' and op != 'n':
        #         print("Invalid option")
        #     op = input("Continue? (y/n) ")
        #     if op == 'n':
        #         sys.exit()



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

    








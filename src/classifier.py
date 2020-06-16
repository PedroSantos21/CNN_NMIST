import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.datasets.mnist import load_data
from keras.optimizers import Adam, Adadelta, Adagrad, Adamax, SGD, RMSprop
from tensorflow.keras import backend as K
import tensorflow as tf

class Classifier:
    
  def __init__(self, train_digits=None, train_labels=None, validation_digits=None, validation_labels=None,test_digits=None,test_labels=None,verbose=1):
    
    self.train_digits       = train_digits
    self.train_labels       = train_labels 
    self.test_digits        = test_digits 
    self.test_labels        = test_labels 
    self.validation_digits  = validation_digits
    self.validation_labels  = validation_labels

    self.verbose    = verbose

    self.numClasses = 10
    
    self.height     = self.train_digits.shape[1]
    self.width      = self.train_digits.shape[2]
    self.channels   = 1

    self.model      = None
   

  def fit(self, test=False, batchSize=64, nbEpochs=10):
    print("Training model...")
    if not test:
      self.model.fit(self.train_digits, 
                    self.train_labels, 
                    batch_size=batchSize, 
                    epochs=nbEpochs,
                    verbose=self.verbose
                  )
    else:
      self.model.fit(self.train_digits, 
                    self.train_labels, 
                    batch_size=batchSize, 
                    epochs=nbEpochs,
                    verbose=self.verbose,
                    validation_data=(self.validation_digits, self.validation_labels)
                  )

  def clear(self):
    K.clear_session()
    # tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()

    
  def evaluate(self, test=False, batchSize=64):
    print("Evaluating model...")
    if not test:
      scores = self.model.evaluate(self.validation_digits, self.validation_labels, batch_size=batchSize, verbose=self.verbose)
    else:
      scores = self.model.evaluate(self.test_digits, self.test_labels, batch_size=batchSize, verbose=self.verbose)
    return dict(zip(self.model.metrics_names, scores))

  # TODO: ESTUDAR OS OTIMIZADORES PARA VER SE SÃO ADEQUADOS
  def checkOptimizer(self, parameters):
    opt = None
    if parameters['optimizer'] == 'Adam':
        opt = Adam(learning_rate=parameters['learningRate'])

    if parameters['optimizer'] == 'Sgd':
        opt = SGD(learning_rate=parameters['learningRate'])
    
    if parameters['optimizer'] == 'RMSprop':
        opt = RMSprop(learning_rate=parameters['learningRate'])
    
    if parameters['optimizer'] == 'Adadelta':
        opt = Adadelta(learning_rate=parameters['learningRate'])
    
    if parameters['optimizer'] == 'Adagrad':
        opt = Adagrad(learning_rate=parameters['learningRate'])
    
    if parameters['optimizer'] == 'Adamax':
        opt = Adamax(learning_rate=parameters['learningRate'])
    
    return opt

  def configureArchitecture(self, parameters):
    self.model = Sequential()
    # CONVOLUTIONAL LAYER 1
    self.model.add(Conv2D(filters=parameters['cnnSize_1'], kernel_size=(3,3), activation='relu', padding='same', input_shape=(self.height, self.width, self.channels)))
    # MAXPOOLING LAYER 1
    self.model.add(MaxPool2D(pool_size=(2,2)))
    # CONVOLUTIONAL LAYER 2
    self.model.add(Conv2D(filters=parameters['cnnSize_2'], kernel_size=(3,3), activation='relu', padding='same'))
    # MAXPOOLING LAYER 2
    self.model.add(MaxPool2D(pool_size=(2,2)))
    # CONVOLUTIONAL LAYER 3
    self.model.add(Conv2D(filters=parameters['cnnSize_3'], kernel_size=(3,3), activation='relu', padding='same'))
    # MAXPOOLING LAYER 3
    self.model.add(MaxPool2D(pool_size=(2,2)))
    self.model.add(Flatten())
    # FULLY CONNECTED LAYERS
    self.model.add(Dense(128, activation='relu'))
    # OUTPUT LAYER
    self.model.add(Dense(self.numClasses, activation='softmax'))

    opt = self.checkOptimizer(parameters)

    self.model.compile( loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])



import numpy as np
from keras.utils import to_categorical
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
from math import *


# AUXILIARY FUNCTIONS
def convertPow2(num):
  '''
    Convert num to the closest power of 2
  '''
  return int(pow(2, ceil(log2(abs(num)))))

def convertRange(num, bounds):
  '''
    Clip number to the bounds
  '''
  num = round(abs(num), num_digits)
  return np.clip(num, *bounds)

def load_dataset():
        (train_digits, train_labels), (test_digits, test_labels) = load_data()
        return (train_digits, train_labels), (test_digits, test_labels)

def load_dataset_with_validation(rate=0.10):
    """
    Load dataset setting apart some validation data
    @args:
        - rate: Percentage of training data to validation
    """

    (train_digits, train_labels), (test_digits, test_labels) = load_dataset()
    
    # RESHAPE DATA
    train_data = reshapeDataset(train_digits)
    test_data  = reshapeDataset(test_digits)

    # RESCALE DATA
    train_data = rescaleDataset(train_data)
    test_data  = rescaleDataset(test_data)

    # ONE-HOT ENCODING
    train_labels_cat = encodingDataset(train_labels)
    test_labels_cat  = encodingDataset(test_labels)

    # SHUFFLE THE TRAINING DATASET
    for _ in range(5):
        indexes = np.random.permutation(len(train_data))
    
    train_data          = train_data[indexes]
    train_labels_cat    =  train_labels_cat[indexes]

    splitPnt = int(rate*len(train_data))

    validation_data         = train_data[:splitPnt,:]
    validation_labels_cat   = train_labels_cat[:splitPnt,:]

    train_data2         = train_data[splitPnt:,:]
    train_labels_cat2   = train_labels_cat[splitPnt:,:]

    return train_data2, train_labels_cat2, test_data, test_labels_cat, validation_data, validation_labels_cat
  
def reshapeDataset(data):
    """
    Reshaping data to CNN standard
    """
    height      = data.shape[1]
    width       = data.shape[2]
    channels    = 1

    return np.reshape(data, (data.shape[0], height, width, channels))

def rescaleDataset(data):
    """
    Rescaling data
    """
    return data.astype('float32')/255

def encodingDataset(dataLabels, numClasses=10):
    """
    ONE-HOT ENCODING
    @args:
        - dataLabels
        - numClasses
    
    @output:
        - List of classes
    """    
    return to_categorical(dataLabels, numClasses)

def showRandomImages(data, labels):
    """
    Exhibit 14 random samples from dataset
    """
    
    np.random.seed(123)

    rand_14 = np.random.randint(0, data.shape[0], 14)
    sample_digits = data[rand_14]
    sample_labels = labels[rand_14]

    num_rows, num_cols = 2,7

    f, ax = plt.subplots(num_rows, num_cols, figsize=(12,5),
                        gridspec_kw={'wspace':0.03, 'hspace':0.01}, 
                        squeeze=True)

    for r in range(num_rows):
        for c in range(num_cols):
            image_index = r * 7 + c
            ax[r,c].axis("off")
            ax[r,c].imshow(sample_digits[image_index], cmap='gray')
            ax[r,c].set_title('No. %d' % sample_labels[image_index])
    plt.show()


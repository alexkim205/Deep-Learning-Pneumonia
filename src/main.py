import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from matplotlib import pyplot as plt
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import cv2
import keras.backend as K

np.random.seed(42)
tf.enable_eager_execution()
save_path = '/home/final/data/cache/{}.bottle_necks.labels.paths.npz'
plot_path = '/home/final/data/plots/bin_classifier'
cache_path = "/home/final/data/cache/"

device = "gpu:0" if tfe.num_gpus() else "cpu:0"


from datamunging import *
from binaryclassifier import *
from multiclassifier import *


def main():
    
    # Fill zipped_data with munged data that has been balanced
    zipped_data[]
    balance_data(zipped_data)
    
    # shuffle
    train_paths, train_labels = paired_shuffle(zipped_data[0])
    validate_paths, validate_labels = paired_shuffle(zipped_data[1])
    test_paths, test_labels = paired_shuffle(zipped_data[2])

    # Save bottle necks for train, validate, and test
    fname = '{}.bottle_necks.labels.paths.npz'

    if not os.path.isdir(cache_path): 
        os.mkdir(cache_path)
    
    train_bottle_necks = cache_bottleneck_layers(train_paths, batch_size=50, device=device)
    np.savez(os.path.join(cache_path,fname.format('train')), bottle_necks=train_bottle_necks, paths=train_paths, labels=train_labels)

    validate_bottle_necks = cache_bottleneck_layers(validate_paths, batch_size=50, device=device)
    np.savez(os.path.join(cache_path,fname.format('validate')), bottle_necks=validate_bottle_necks, paths=validate_paths, labels=validate_labels)

    test_bottle_necks = cache_bottleneck_layers(test_paths, batch_size=50, device=device)
    np.savez(os.path.join(cache_path,fname.format('test')), bottle_necks=test_bottle_necks, paths=test_paths, labels=test_labels)
    
    #####################
    # BINARY CLASSIFIER #
    #####################
    
    # Read in binary classification data
    train_dataset_original, validate_dataset_original, test_dataset_original = load_bin_data(save_path)
    
    # Parameters to try out
    list_n_layers = [3] * 7 + [6] * 7 + [9] * 7
    list_batch_size = [64, 32] * 3
    list_n_epochs = [50, 50, 50, 50, 50, 100, 200] * 3
    list_learning_rate = [0.01, 0.01, 0.1, 0.001, 0.0001, 0.01, 0.01] * 3
    n_classes = 2
    
    # Train models
    for i, (n_layers, batch_size, n_epochs, learning_rate) in \
        enumerate(zip(list_n_layers, list_batch_size, list_n_epochs, list_learning_rate), 1):

        save_plot_fname = "L{}.BS{}.LR{}.EP{}".format(n_layers, batch_size, learning_rate, n_epochs)
        title = "| Model number {:02d}: {} |".format(i, save_plot_fname)
        print("-"*len(title))
        print(title)
        print("-"*len(title))
    
        train_dataset = train_dataset_original.shuffle(buffer_size=50).batch(batch_size)
        validate_dataset = validate_dataset_original.shuffle(buffer_size=50).batch(batch_size)
        test_dataset = validate_dataset_original.shuffle(buffer_size=50).batch(batch_size)

        x_classifier, tl, vl, ta, va, tf1, vf1 = train(train_dataset, validate_dataset, learning_rate, batch_size, n_epochs, n_layers, n_classes, save_plot_fname)
        test(x_classifier, test_dataset, save_plot_fname, n_classes)
        
    ####################
    # MULTI-CLASSIFIER #
    ####################
    
    # Read in binary classification data
    train_dataset_original, validate_dataset_original, test_dataset_original = load_bin_data(save_path)
    
    # Parameters to try out
    list_n_layers = [3] * 7 + [6] * 7 + [9] * 7
    list_batch_size = [64, 32] * 3
    list_n_epochs = [50, 50, 50, 50, 50, 100, 200] * 3
    list_learning_rate = [0.01, 0.01, 0.1, 0.001, 0.0001, 0.01, 0.01] * 3
    n_classes = 3
    
    # Train models
    for i, (n_layers, batch_size, n_epochs, learning_rate) in \
        enumerate(zip(list_n_layers, list_batch_size, list_n_epochs, list_learning_rate), 1):

        save_plot_fname = "L{}.BS{}.LR{}.EP{}".format(n_layers, batch_size, learning_rate, n_epochs)
        title = "| Model number {:02d}: {} |".format(i, save_plot_fname)
        print("-"*len(title))
        print(title)
        print("-"*len(title))
    
        train_dataset = train_dataset_original.shuffle(buffer_size=50).batch(batch_size)
        validate_dataset = validate_dataset_original.shuffle(buffer_size=50).batch(batch_size)
        test_dataset = validate_dataset_original.shuffle(buffer_size=50).batch(batch_size)

        x_classifier, tl, vl, ta, va, tf1, vf1 = train(train_dataset, validate_dataset, learning_rate, batch_size, n_epochs, n_layers, n_classes, save_plot_fname)
        test(x_classifier, test_dataset, save_plot_fname)
        
    
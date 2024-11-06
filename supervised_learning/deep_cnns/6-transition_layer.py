#!/usr/bin/env python3
"""
6. Transition Layer"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """builds a transition layer
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer
    Your code should implement compression as used in DenseNet-C
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    All convolutions should be preceded by Batch Normalization and a
    __rectified linear activation (ReLU), respectively
    Returns: The output of the transition layer and the number of filters
    __within the output, respectively"""
    initializer = K.initializers.he_normal(seed=0)
    nb_filters = int(nb_filters * compression)
    batch_normalization = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_normalization)
    conv2d = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=1,
        padding='same',
        kernel_initializer=initializer
    )(activation)
    average_pooling2d = K.layers.AveragePooling2D(
        pool_size=2,
        padding='same'
    )(conv2d)
    return average_pooling2d, nb_filters

#!/usr/bin/env python3
""" build NN with keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """nx is the number of input features to the network
    layers is a list of nodes in each layer
    activations list of activations for each layer
    lambtha is the L2 regularization parameter
    keep_prob proba that a node will be kept for dropout
    """
    model = K.Sequential()
    regularizer = K.regularizers.L2(lambtha)
    model.add(K.layers.Dense(
            layers[0], activation=activations[0],
            kernel_regularizer=regularizer,
            input_shape=(nx,)))
    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=regularizer))

    return model

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
    x = K.Input(shape=(nx,))
    regularizer = K.regularizers.L2(lambtha)
    layer = K.layers.Dense(
            layers[0], activation=activations[0],
            kernel_regularizer=regularizer)(x)
    for i in range(1, len(layers)):
        layer = K.layers.Dropout(1 - keep_prob)(layer)
        layer = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=regularizer)(layer)

    model = K.Model(inputs=x, outputs=layer)
    return model

#!/usr/bin/env python3
"""miugljygjlg"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

create_layer = __import__('1-create_layer').create_layer

def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for a neural network

    Arguments:
    x -- placeholder for the input data
    layer_sizes -- list containing the number of nodes in each layer of the network
    activations -- list containing the activation functions for each layer of the network

    Returns:
    prediction of the network in tensor form
    """
    A = x

    for i in range(len(layer_sizes)):
        A_prev = A
        A = create_layer(A_prev, layer_sizes[i], activations[i])

    Y_pred = A

    return Y_pred

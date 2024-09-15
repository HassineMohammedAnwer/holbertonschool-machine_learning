#!/usr/bin/env python3
"""afml aefo ^pkeaf """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conduct forward propagation with dropout.

    Parameters:
        X: numpy.ndarray, input data (nx, m).
        weights: dictionary, weights and biases of the neural network.
        L: int, number of layers in the network.
        keep_prob: float, probability that a node will be kept during dropout.

    Returns:
        cache: dictionary, outputs of each layer and dropout masks.
    """
    m = X.shape[1]
    cache = {'A0': X}

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        Z = np.dot(W, A_prev) + b
        if i == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = (np.random.rand(A.shape[0], A.shape[1]) <
                 keep_prob).astype(int)
            A *= D
            A /= keep_prob
            cache['D' + str(i)] = D
        cache['A' + str(i)] = A

    return cache

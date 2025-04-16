#!/usr/bin/env python3
"""
4. Deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """performs forward propagation for a deep RNN:
    rnn_cells is a list of RNNCell instances of length
    __l that will be used for the forward propagation
    l is the number of layers
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 :the initial hidden state, given as a numpy.ndarray of shape (l, m, h)
    h is the dimensionality of the hidden state
    Returns: H, Y
    H is a numpy.ndarray containing all of the hidden states
    Y is a numpy.ndarray containing all of the outputs
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, rnn_cells[-1].by.shape[1]))
    for i in range(t):
        layer_input = X[i]
        for layer, cell in enumerate(rnn_cells):
            if layer == 0:
                h_next, y = cell.forward(H[i, layer], X[i])
            else:
                h_next, y = cell.forward(H[i, layer], h_next)
            H[i + 1, layer] = h_next
        Y[i] = y

    return H, Y

#!/usr/bin/env python3
"""performs forward propagation for a simple RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """rnn_cell:instance of RNNCell
    Returns: H, Y
    H:ndarray containing all of the hidden states
    Y:ndarray containing all of the outputs"""
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]
    # Initialize an array to store hidden states + other for output
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    # Set the initial hidden state.
    H[0] = h_0
    for i in range(t):
        # Select the input at time step i.
        x_t = X[i]
        # Select the previous hidden state.
        h_prev = H[i]
        # Perform forward propagation.
        h_next, y_next = rnn_cell.forward(h_prev=h_prev, x_t=x_t)
        # Store the next hidden state. and output
        H[i + 1] = h_next
        Y[i] = y_next

    return H, Y

#!/usr/bin/env python3
"""recurrent neural networks"""
import numpy as np


class RNNCell:
    """represents a cell of a simple RNN"""
    def __init__(self, i, h, o):
        """class constructor with:
        i = dimensionality of the data
        h = dimensionality of the hidden state
        o = dimensionality of the outputs"""

        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward propagation for one time step
        x_t:contains the data input for the cell
        m is the batch size for the data
        h_t:containing the previous hidden state
        The output of the cell should use a softmax activation function
        h_next is the next hidden state
        y is the output of the cell"""
        #calculate new hidden state
        x_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.matmul(x_concat, self.Wh) + self.bh
        h_next = np.tanh(h_next)
        y = np.matmul(h_next, self.Wy) + self.by
        #softmax
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return (h_next, y)

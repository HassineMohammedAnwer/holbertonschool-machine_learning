#!/usr/bin/env python3
""" BidirectionalCell backward
"""

import numpy as np


class BidirectionalCell:
    """jmlkhùlk"""
    def __init__(self, i, h, o):
        """constructor"""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward"""
        inp = np.concatenate((h_prev, x_t), 1)
        h_next = np.tanh((inp @ self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """the previous hidden state.
        """
        inp = np.concatenate((h_next, x_t), 1)
        # Apply the hyperbolic tangent activation function
        z = np.tanh(inp @ self.Whb + self.bhb)
        return z

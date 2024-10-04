#!/usr/bin/env python3
"""
    One-Hot Encode
"""


import numpy as np


def one_hot_encode(Y, classes):
    """ converts a numerical label vector into a one-hot vector"""
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < np.max(Y) + 1:
        return None

    # create 2d matrix filled with zeros
    one_hot_marix = np.zeros((classes, Y.size), dtype=float)

    # replacing 0 with a 1 at the index of the original array
    one_hot_marix[Y, np.arange(Y.size)] = 1

    return one_hot_marix

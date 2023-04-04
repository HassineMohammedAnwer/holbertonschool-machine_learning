#!/usr/bin/env python3
"""Shuffles"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
    - X (numpy.ndarray): The first numpy.ndarray of shape (m, nx) to shuffle.
    - Y (numpy.ndarray): The second numpy.ndarray of shape (m, ny) to shuffle.

    Returns:
    A tuple containing the shuffled X and Y matrices.
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    return X_shuffled, Y_shuffled

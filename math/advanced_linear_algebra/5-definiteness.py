#!/usr/bin/env python3
""" task 5. Definiteness """
import numpy as np


def definiteness(matrix):
    """calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if not np.array_equal(matrix, matrix.T):
        return None
    n = len(matrix)
    if n == 0:
        return None
    if any(len(row) != n for row in matrix):
        raise None

    eigen_values, _ = np.linalg.eig(matrix)
    min = np.min(eigen_values)
    max = np.max(eigen_values)
    if min > 0 and max > 0:
        return "Positive definite"
    elif min == 0 and max > 0:
        return "Positive semi-definite"
    elif min < 0 and max < 0:
        return "Negative definite"
    elif min < 0 and max == 0:
        return "Negative semi-definite"
    elif min < 0 < max:
        return "Indefinite"
    else:
        return None

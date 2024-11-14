#!/usr/bin/env python3
"""1. Regular Chains"""
import numpy as np


def regular(P):
    """determines the steady state probabilities of a
    __regular markov chain:
    P is a is a square 2D numpy.ndarray of shape (n, n)
    __representing the transition matrix
    P[i, j] is the probability of transitioning from state i to state j
    n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady
    __state probabilities, or None on failure"""

    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1] or P.ndim != 2:
        return None

    if not np.allclose(P.sum(axis=1), 1):
        return None
    n = P.shape[0]
    A = P.T - np.eye(n)
    b = np.ones(n)
    try:
        station_M = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    # Normalize the solution vector to ensure the sum is 1
    station_M = station_M / station_M.sum()

    return station_M.reshape(1, n)
#!/usr/bin/env python3
"""2. Absorbing Chains"""
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing:
    P is a is a square 2D numpy.ndarray of shape (n, n) representing
    __the standard transition matrix
    P[i, j] is the probability of transitioning from state i to state j
    n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure"""
    if not isinstance(P,
                      np.ndarray) or P.shape[0] != P.shape[1] or P.ndim != 2:
        return False

    if not np.allclose(P.sum(axis=1), 1):
        return False
    A = np.zeros(P.shape[0], dtype=bool)

    while True:
        prev = A.copy()
        A = np.any(P == 1, axis=0)
        if np.all(A == prev):
            return False
        if A.all():
            return True
        r = np.any(P[:, A], axis=1)
        P[r, r] = 1

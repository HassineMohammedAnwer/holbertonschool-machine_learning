#!/usr/bin/env python3
"""Marginal Probability"""
import numpy as np


def likelihood(x, n, P):
    """calculates the likelihood of obtaining
    this data given various hypothetical probabilities
    of developing severe side effects"""    
    n_fa = 1
    for i in range(1, n + 1):
        n_fa *= i
    x_fa = 1
    for i in range(1, x + 1):
        x_fa *= i
    n_s_x_fa = 1
    for i in range(1, (n - x) + 1):
        n_s_x_fa *= i
    b_coefficient = n_fa / (x_fa * n_s_x_fa)
    likelihoods = np.zeros_like(P)
    for i, p in enumerate(P):
        likelihoods[i] = b_coefficient * (p ** x) * ((1 - p) ** (n - x))
    return likelihoods

def intersection(x, n, P, Pr):
    """intersection = likelihood * prior"""
    likelihoods = likelihood(x, n, P)

    intersections = likelihoods * Pr

    return intersections

def marginal(x, n, P, Pr):
    """intersection = likelihood * prior"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater\
                         than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    marginal_prob = np.sum(intersection(x, n, P, Pr))
    return marginal_prob
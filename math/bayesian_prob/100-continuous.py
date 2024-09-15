#!/usr/bin/env python3
"""Marginal Probability"""
import numpy as np
from scipy import special


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
    """marginal"""
    marginal_prob = np.sum(intersection(x, n, P, Pr))
    return marginal_prob


def posterior(x, n, P, Pr):
    """ posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data"""
    posterior_prob = intersection(x, n, P, Pr) / marginal(x, n, P, Pr)
    return posterior_prob


def posterior(x, n, p1, p2):
    """posterior probability that the probability of developing severe
    side effects falls within a specific range given the data"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer \
that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    alpha = x + 1
    beta = n - x + 1
    beta_cdf_p1 = special.betainc(alpha, beta, p1)
    beta_cdf_p2 = special.betainc(alpha, beta, p2)
    return beta_cdf_p2 - beta_cdf_p1

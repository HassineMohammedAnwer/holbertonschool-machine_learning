#!/usr/bin/env python3
"""0. Markov Chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """determines the probability of a markov chain being in
    __a particular state after a specified number of iterations:
    P is a square 2D numpy.ndarray of shape (n, n) representing
    __the transition matrix
    P[i, j] is the probability of transitioning from state i to state j
    n is the number of states in the markov chain
    s is a numpy.ndarray of shape (1, n) representing the probability
    __of starting in each state
    t is the number of iterations that the markov chain has been through
    Returns: a numpy.ndarray of shape (1, n) representing the probability
    __of being in a specific state after t iterations, or None on failure"""
    current_state = s
    for i in range(t):
        next_state = np.matmul(current_state, P)
        current_state = next_state
    return current_state

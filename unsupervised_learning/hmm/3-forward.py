#!/usr/bin/env python3
"""3. The Forward Algorithm"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model:
    Observation is a numpy.ndarray of shape (T,) that contains the
    __index of the observation
    T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission
    __probability of a specific observation given a hidden state
    Emission[i, j] is the probability of observing j given the hidden state i
    N is the number of hidden states
    M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the
    __transition probabilities
    Transition[i, j] is the probability of transitioning from the
    __hidden state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability
    __of starting in a particular hidden state
    Returns: P, F, or None, None on failure
    P is the likelihood of the observations given the model
    F is a numpy.ndarray of shape (N, T) containing the forward
    __path probabilities
    F[i, j] is the probability of being in hidden state i at
    __time j given the previous observations"""
    if type(Observation) != np.ndarray:
        return None, None
    T = Observation.shape[0]

    if type(Emission) != np.ndarray:
        return None, None
    N = Emission.shape[0]

    if type(Transition) != np.ndarray:
        return None, None

    if type(Initial) != np.ndarray:
        return None, None

    F = np.zeros((N, T))
    F[:, 0] = np.multiply(Initial.T, Emission[:, Observation[0]])

    for i in range(1, T):
        for j in range(N):

            SDD = Transition[:, j]
            HDD = Emission[j, Observation[i]]
            F[j, i] = np.sum(F[:, i - 1] * HDD * SDD)

    P = F[:, -1:].sum()
    return P, F

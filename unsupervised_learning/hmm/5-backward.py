#!/usr/bin/env python3
"""5. The Backward Algorithm"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """performs the backward algorithm for a hidden markov model:
    Observation is a numpy.ndarray of shape
    __(T,) that contains the index of the observation
    T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the
    __emission probability of a specific observation given a hidden state
    Emission[i, j] is the probability of observing j given the hidden state i
    N is the number of hidden states
    M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N)
    __containing the transition probabilities
    Transition[i, j] is the probability of transitioning from
    __the hidden state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the
    __probability of starting in a particular hidden state
    Returns: P, B, or None, None on failure
    P is the likelihood of the observations given the model
    B is a numpy.ndarray of shape (N, T) containing the
    __backward path probabilities
    B[i, j] is the probability of generating the future
    __observations from hidden state i at time j"""
    if (not isinstance(Observation, np.ndarray) or Observation.ndim != 1 or
            not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
            not isinstance(Transition, np.ndarray) or Transition.ndim != 2 or
            not isinstance(Initial, np.ndarray) or Initial.ndim != 2):
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    N, M = Emission.shape
    if Transition.shape != (N, N) or Initial.shape[0] != N:
        return None, None
    T = Observation.shape[0]
    B = np.zeros((N, T))
    B[:, T-1] = 1.0
    for t in range(T-2, -1, -1):
        next_obs = Observation[t + 1]
        emission_probs = Emission[:, next_obs]
        B[:, t] = np.dot(Transition, emission_probs * B[:, t + 1])
    first_obs = Observation[0]
    initial_probs = Initial.ravel()
    emission_init = Emission[:, first_obs]
    P = np.sum(initial_probs * emission_init * B[:, 0])

    return P, B

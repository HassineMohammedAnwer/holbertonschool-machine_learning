#!/usr/bin/env python3
"""4. The Viretbi Algorithm"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """calculates the most likely sequence of hidden
    __states for a hidden markov model:
    Observation is a numpy.ndarray of shape (T,)
    ___that contains the index of the observation
    T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the
    ___emission probability of a specific observation given a hidden state
    Emission[i, j] is the probability of observing j given the hidden state i
    N is the number of hidden states
    M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N)
    ___containing the transition probabilities
    Transition[i, j] is the probability of
    ___transitioning from the hidden state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the
    ___probability of starting in a particular hidden state
    Returns: path, P, or None, None on failure
    path is the a list of length T containing the most likely
    ___sequence of hidden states
    P is the probability of obtaining the path sequence"""
    if (not isinstance(Observation, np.ndarray) or Observation.ndim != 1 or
            not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
            not isinstance(Transition, np.ndarray) or Transition.ndim != 2 or
            not isinstance(Initial, np.ndarray) or Initial.ndim != 2):
        return None, None
    N, M = Emission.shape
    if Transition.shape != (N, N) or Initial.shape[0] != N:
        return None, None
    T = Observation.shape[0]
    V = np.zeros((N, T))
    b_pointer = np.zeros((N, T), dtype=int)
    first_obs = Observation[0]
    V[:, 0] = (Initial[:, 0] * Emission[:, first_obs])
    for t in range(1, T):
        for j in range(N):
            trans_probs = V[:, t-1] * Transition[:, j]
            max_idx = np.argmax(trans_probs)
            max_val = trans_probs[max_idx]
            V[j, t] = max_val * Emission[j, Observation[t]]
            b_pointer[j, t] = max_idx
    P = np.max(V[:, -1])
    if P <= 0:
        return None, None
    last = np.argmax(V[:, -1])
    path = [last]
    for t in reversed(range(1, T)):
        prev_state = b_pointer[path[-1], t]
        path.append(prev_state)
    path = path[::-1]

    return path, P

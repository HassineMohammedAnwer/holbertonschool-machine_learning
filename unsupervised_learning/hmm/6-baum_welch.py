#!/usr/bin/env python3
"""6. The Baum-Welch Algorithm"""
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
    if type(Observation) is not np.ndarray:
        return None, None
    T = Observation.shape[0]

    if type(Emission) is not np.ndarray:
        return None, None
    N = Emission.shape[0]

    if type(Transition) is not np.ndarray:
        return None, None

    if type(Initial) is not np.ndarray:
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

def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """performs the Baum-Welch algorithm for a hidden markov model:
    Observations is a numpy.ndarray of shape (T,)
    __that contains the index of the observation
    T is the number of observations
    Transition is a numpy.ndarray of shape (M, M) that
    __contains the initialized transition probabilities
    M is the number of hidden states
    Emission is a numpy.ndarray of shape (M, N) that contains
    __the initialized emission probabilities
    N is the number of output states
    Initial is a numpy.ndarray of shape (M, 1) that contains
    __the initialized starting probabilities
    iterations is the number of times expectation-maximization
    __should be performed
    Returns: the converged Transition, Emission, or None,
    __None on failure"""
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    N = Transition.shape[0]
    M = Emission.shape[1]
    T = Observations.shape[0]
    for _ in range(iterations):
        P_f, F = forward(Observations, Emission, Transition, Initial)
        scaling = np.zeros(T)
        scaling[0] = 1.0 / (F[:, 0].sum() + 1e-10)
        for t in range(1, T):
            scaling[t] = 1.0 / (F[:, t].sum() + 1e-10)
        p_b, B = backward(Observations, Emission, Transition, scaling)
        if P_f <= 0:
            return None, None
        xi = F[:, None, :-1] * Transition[:, :, None] * Emission[:, Observations[1:]][None, :, :] * B[None, :, 1:]
        xi /= P_f
        xi_sum = xi.sum(axis=(0, 1), keepdims=True)
        xi_sum[xi_sum == 0] = 1e-10
        xi /= xi_sum
        gamma = np.sum(xi, axis=1)
        gamma_last = (F[:, -1] * B[:, -1]).reshape(-1, 1)
        gamma_last_sum = gamma_last.sum()
        gamma_last = gamma_last / gamma_last_sum if gamma_last_sum != 0 else gamma_last
        gamma = np.hstack((gamma, gamma_last))
        sum_xi = np.sum(xi, axis=2)
        sum_gamma = np.sum(gamma[:, :-1], axis=1, keepdims=True)
        sum_gamma[sum_gamma == 0] = 1e-10
        Transition = sum_xi / sum_gamma
        mask = (Observations[:, np.newaxis] == np.arange(M))
        sum_emit = gamma @ mask
        sum_gamma_total = np.sum(gamma, axis=1, keepdims=True)
        sum_gamma_total[sum_gamma_total == 0] = 1e-10
        Emission = sum_emit / sum_gamma_total

    return Transition, Emission

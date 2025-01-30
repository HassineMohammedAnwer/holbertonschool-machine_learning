#!/usr/bin/env python3
"""0. Simple Policy function
&
1. Compute the Monte-Carlo policy gradient"""
import numpy as np


def policy(matrix, weight):
    """Computes the policy with a weight matrix"""
    z = np.dot(matrix, weight)
    tmp = np.exp(z - np.max(z))
    prob = tmp / np.sum(tmp, axis=1, keepdims=True)
    return prob


def policy_gradient(state, weight):
    """computes the Monte-Carlo policy gradient
    based on a state and a weight matrix.
    state: matrix representing the current observation
    of the environment
    weight: matrix of random weight
    Return: the action and the gradient (in this order)"""
    state = state.reshape(1, -1)
    prob = policy(state, weight)
    action = np.random.choice(prob.shape[1], p=prob.ravel())
    grad = np.zeros_like(weight)
    for i in range(weight.shape[1]):
        grad[:, i] = state * ((1 if i == action else 0) - prob[0, i])
    return action, grad

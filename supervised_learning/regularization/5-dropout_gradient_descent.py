#!/usr/bin/env python3
"""pdates the weights and biases dropout """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ Y= labels
    weights = dictionnary of weights
    cache= dictionary of outputs and dropout masks of each layer
    alpha= learning rate
    keep_prob= proba that node'll be kept
    L= num of layers"""
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]

        dW = (1 / m) * np.dot(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        # for all layers except the output layer
        if i > 1:
            dA_prev = np.dot(W.T, dz)
            D = cache['D' + str(i - 1)]
            dA_prev = np.multiply(dA_prev, D)
            dA_prev /= keep_prob

            dz = dA_prev * (1 - A_prev ** 2)

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

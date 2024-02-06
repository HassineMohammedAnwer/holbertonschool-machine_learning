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
    # Backward propagation
    for i in range(L - 1, -1, -1):
        A_prev = cache['A' + str(i)]
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        A = cache['A' + str(i + 1)]
        if i == L - 1:
            rs = A - Y
        else:
            D = cache['D' + str(i + 1)]
            rs = np.matmul(weights['W' + str(i + 1)].T, rs) * \
                (1 - (A * A)) * D / keep_prob
        dW = np.matmul(rs, A_prev.T) / m
        db = np.sum(rs, axis=1, keepdims=True) / m
        weights['W' + str(i)] = weights["W" + str(i)] - (alpha * dW)
        weights['b' + str(i)] = weights["b" + str(i)] - (alpha * db)

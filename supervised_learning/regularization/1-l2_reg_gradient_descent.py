#!/usr/bin/env python3
"""pdates the weights and biases L2 """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """moij ossdcz ref mijhuomh"""
    # Number of training examples
    m = Y.shape[1]
    # Compute the output layer's delta (error)
    tmp = cache['A' + str(L)]
    dZ = tmp - Y
    # Backward pass through the layers
    for i in range(L, 0, -1):
        # Get the output of the previous layer
        A_prev = cache['A' + str(i - 1)]
        # Calculate the weight gradient with L2 regularization
        dW = (1 / m) * np.dot(dZ, A_prev.T) +
            (lambtha / m) * weights['W' + str(i)]
        # Calculate the bias gradient
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        # Calculate delta (error) for the previous layer
        dZ = np.dot(weights['W' + str(i)].T, dZ) * (1 - np.power(A_prev, 2))

        # Update the weights using the gradient descent update rule
        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dW
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db

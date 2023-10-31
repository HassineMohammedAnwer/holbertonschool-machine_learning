#!/usr/bin/env python3
"""lk,ùlkn"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """cal the cost of a neural network with L2 regularization"""
    sum = 0
    for i in range(1, L + 1):
        w = weights['W{}'.format(i)]
        sum += np.sum(w ** 2)
    return cost + lambtha * (1 / (2 * m)) * sum

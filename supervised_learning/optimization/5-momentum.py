#!/usr/bin/env python3
"""5. Momentum"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    updates variable using gradient descent with momentum optimization algo
    """
    var_new = beta1 * v + (1 - beta1) * grad
    new_var = var - var_new * alpha
    return new_var, var_new

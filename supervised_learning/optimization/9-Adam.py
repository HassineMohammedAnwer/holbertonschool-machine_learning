#!/usr/bin/env python3
"""9. Adam"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    updates a variable in place using the Adam optimization algo
    var: W or b
    grad: dW or db
    s: dW_prev or db_prev
    """
    var_new = beta1 * v + (1 - beta1) * grad
    new_s = (beta2 * s) + (1 - beta2) * (grad ** 2)
    corctn_var_new = var_new / (1 - beta1 ** t)
    corctn_new_s = new_s / (1 - beta2 ** t)
    new_var = var - (alpha * (corctn_var_new / ((corctn_new_s ** 0.5) + epsilon)))
    return new_var, var_new, new_s

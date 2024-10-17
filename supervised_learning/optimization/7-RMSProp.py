#!/usr/bin/env python3
"""7. RMSProp"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using the RMSProp optimization algo
    var: W or b
    grad: dW or db
    s: dW_prev or db_prev
    """
    new_s = (beta2 * s) + (1 - beta2) * (grad ** 2)
    new_var = var - (alpha * (grad / ((new_s ** 0.5) + epsilon)))
    return new_var, new_s

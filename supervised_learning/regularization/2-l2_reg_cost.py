#!/usr/bin/env python3
"""cost of a neural network with L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost):
    """mljlj poiùp ùpi"""
    cost += tf.losses.get_regularization_losses()
    return cost

#!/usr/bin/env python3
"""cost of a neural network with L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """mljlj poiùp ùpi"""
    l2_reg_loss = list()
    for layer in model.layers:
        l2_reg_loss.append(tf.reduce_sum(layer.losses) + cost)

    return tf.stack(l2_reg_loss[1:])
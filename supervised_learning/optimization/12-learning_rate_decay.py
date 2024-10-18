#!/usr/bin/env python3
"""12. Learning Rate Decay Upgraded"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """updates learning rate using inverse time decay in numpy"""
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)

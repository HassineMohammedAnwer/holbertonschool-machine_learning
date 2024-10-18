#!/usr/bin/env python3
"""11. Learning Rate Decay"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """updates learning rate using inverse time decay in numpy"""
    return alpha / (1 + decay_rate * (global_step // decay_step))

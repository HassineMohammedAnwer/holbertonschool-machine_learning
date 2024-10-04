#!/usr/bin/env python3
"""init q-table of zeros"""
import numpy as np


def q_init(env):
    """takes num of all states of env as col
    and actions as rows"""
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    q_table = np.zeros((state_space_size, action_space_size))
    return q_table

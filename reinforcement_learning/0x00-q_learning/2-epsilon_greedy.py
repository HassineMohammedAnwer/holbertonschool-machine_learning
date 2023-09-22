#!/usr/bin/env python3
"""exploitation or exploration"""

import numpy as np

def epsilon_greedy(Q, state, epsilon):
    """uses epsilon-greedy to determine the next action
    """

    # random_num
    p = np.random.uniform(0, 1)
    if p >= epsilon:
        # choose action via exploitation
        action = np.argmax(Q[state])
    else:
        # choose action via exploration
        action = np.random.randint(Q.shape[1])
    return action

#!/usr/bin/env python3
"""exploitation or exploration"""

import numpy as np


def play(env, Q, max_steps=100):
    """k,p,p,ùp,"""
    state_tuple = env.reset()
    state = state_tuple[0]
    total_rewards = 0
    print(env.render(), end='')

    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, _t, _i = env.step(action)

        # Render the environment (no mode argument needed)
        print(env.render(), end='')

        total_rewards += reward
        state = new_state

        if done:
            break
    return total_rewards

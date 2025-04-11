#!/usr/bin/env python3
"""exploitation or exploration"""

import numpy as np


def play(env, Q, max_steps=100):
    """k,p,p,ùp,"""
    state_tuple = env.reset()
    state = state_tuple[0]
    total_reward = 0
    rendered_outputs = []

    for step in range(max_steps):
        rendered = env.render()
        rendered_outputs.append(rendered)
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        action_names = ['Left', 'Down', 'Right', 'Up']
        rendered_outputs.append(f'  ({action_names[action]})')
        state = next_state
        if terminated or truncated:
            rendered_outputs.append(env.render())
            break
    return total_reward, rendered_outputs

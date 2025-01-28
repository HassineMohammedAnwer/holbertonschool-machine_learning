#!/usr/bin/env python3
"""1. TD(λ)"""
import numpy as np


def td_lambtha(env, V, policy, lambtha,
               episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """performs the TD(λ) algorithm:
    env is the environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    policy is a function that takes in a state and
    returns the next action to take
    lambtha is the eligibility trace factor
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    Returns: V, the updated value estimate"""
    states = V.shape[0]
    for _ in range(episodes):
        state = env.reset()[0]
        r = np.zeros(states)
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            delta = reward + (gamma * V[next_state]) - V[state]
            r[state] += 1
            V = V + alpha * delta * r
            r *= gamma * lambtha
            state = next_state
            if done:
                break
    return V

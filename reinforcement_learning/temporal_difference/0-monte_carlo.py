#!/usr/bin/env python3
"""0. Monte Carlo"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm:
    env is environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    policy:function: takes in a state and returns the next action to take
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    Returns: V, the updated value estimate
    """
    for ep in range(episodes):
        episode = []
        state = env.reset()[0]
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            episode.append([state, action, reward, next_state])
            if done:
                break
            state = next_state
        episode = np.array(episode, dtype=int)
        G = 0
        for _, step in enumerate(episode[::-1]):
            state, action, reward, _ = step
            G = gamma * G + reward
            if state not in episode[:ep, 0]:
                V[state] = V[state] + alpha * (G - V[state])
    return V

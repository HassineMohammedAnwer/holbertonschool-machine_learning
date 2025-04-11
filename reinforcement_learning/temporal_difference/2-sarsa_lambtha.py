#!/usr/bin/env python3
"""
lkh
"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    performs SARSA(λ):
    env is the environment instance
    Q is a numpy.ndarray of shape (s,a) containing the Q table
    lambtha is the eligibility trace factor
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    epsilon is the initial threshold for epsilon greedy
    min_epsilon is the minimum value that epsilon should decay to
    epsilon_decay is the decay rate for updating epsilon between episodes
    Returns: Q, the updated Q table
    """
    nS, nA = Q.shape
    initial_epsilon = epsilon

    def select_action(state, epsilon):
        """
        Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(nA)
        return np.argmax(Q[state])

    for episode in range(episodes):
        E = np.zeros((nS, nA))
        state, _ = env.reset()
        action = select_action(state, epsilon)
        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = select_action(next_state, epsilon)
            td_error = reward + gamma * Q[next_state,
                                          next_action] - Q[state, action]
            E[state, action] += 1
            Q += alpha * td_error * E
            E *= gamma * lambtha
            state, action = next_state, next_action
            if terminated or truncated:
                break
        epsilon = min_epsilon + (
            initial_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
    return Q

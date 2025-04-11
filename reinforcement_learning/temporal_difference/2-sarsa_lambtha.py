#!/usr/bin/env python3
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
    for episode in range(episodes):
        E = np.zeros_like(Q)
        state = env.reset()[0]
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        for _ in range(max_steps):
            next_state, reward, done, truncated, _ = env.step(action)
            if np.random.uniform() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            td_error = reward + gamma * Q[next_state][next_action] - Q[state][action]
            E[state][action] += 1
            Q += alpha * td_error * E
            E *= gamma * lambtha
            state, action = next_state, next_action
            if done or truncated:
                break
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
    return Q

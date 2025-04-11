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
    E = np.zeros_like(Q)
    for episode in range(episodes):
        E.fill(0)
        state = env.reset()[0]
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])
        step = 0
        done = False
        truncated = False

        while not (done or truncated) and step < max_steps:
            next_state, reward, done, truncated, _ = env.step(action)
            if np.random.random() < epsilon:
                next_action = np.random.randint(n_actions)
            else:
                next_action = np.argmax(Q[next_state])
            td_error = reward + gamma * Q[next_state, next_action] - Q[state, action]
            E[state, action] += 1
            Q += alpha * td_error * E
            E *= gamma * lambtha
            state, action = next_state, next_action
            step += 1
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
    return Q

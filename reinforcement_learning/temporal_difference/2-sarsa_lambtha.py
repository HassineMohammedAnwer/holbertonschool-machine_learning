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
    for _ in range(episodes):
        E.fill(0)
        done = False
        truncated = False
        state = env.reset()[0]
        action = get_action(state, Q, epsilon)

        while not done:
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = get_action(next_state, Q, epsilon)

            target = reward + gamma * Q[next_state, next_action]
            actual = Q[state, action]
            delta = target - actual

            E[state, action] += 1
            Q += alpha * delta * E
            E *= gamma * lambtha

            state, action = next_state, next_action
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
    return Q


def get_action(state, Q, epsilon):
    """
    Epsilon-greedy action selection    """
    n_actions = Q.shape[1]
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])

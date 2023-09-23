#!/usr/bin/env python3
"""qtrain"""

import numpy as np
import gym
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy

def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """train ql
    """
    rewards_all_episodes = []

    for episode in range(episodes):
        state_tuple = env.reset()
        state = state_tuple[0]  # Extract the initial state from the tuple
        state = int(state)  # Cast state to integer
        done = False
        rewards_current_episode = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, _, _ = env.step(action)

            if done and reward == 0:
                reward = -1

            state = int(state)  # Cast state to integer
            action = int(action)  # Cast action to integer

            Q[state][action] = Q[state][action] * (1 - alpha) + alpha * (reward + gamma * np.max(Q[new_state]))

            state = new_state
            rewards_current_episode += reward

            if done:
                break

        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))
        rewards_all_episodes.append(rewards_current_episode)

    return Q, rewards_all_episodes

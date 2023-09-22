
#!/usr/bin/env python3
"""exploitation or exploration"""

import numpy as np
import gym
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy

def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """uses epsilon-greedy to determine the next action
    """
    total_rewards = []
    # eoe
    for episode in range(episodes):
        state = env.reset()
        done = False
        rewards_current_episode = 0

        for step in range(max_steps):
            # Exploration-exploitation trade-off
            action = epsilon_greedy(Q, state, epsilon)
            # take action
            new_state, reward, done, truncated, info = env.step(action)
            if done and reward == 0:
                # Update the reward to -1
                reward = -1
            rewards_current_episode += 1
            # Update The Q-Value
            Q[state, action] = Q[state, action] * (1 - alpha) + alpha * (reward + gamma * np.max(Q[new_state]))
            # Transition To The Next State
            state = new_state
            if done == True:
                break
        # Exploration rate decay
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
        total_rewards.append(rewards_current_episode)
    return Q, total_rewards

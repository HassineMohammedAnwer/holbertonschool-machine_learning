#!/usr/bin/env python3
"""2. Implement the training"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """ implements a full training.
    env: initial environment
    nb_episodes: number of episodes used for training
    alpha: the learning rate
    gamma: the discount factor
    You should use policy_gradient = __import__('policy_gradient').policy_gradient
    Return: all values of the score (sum of all rewards during one episode loop)
    You need print the current episode number and the score after each loop
    in a format: Episode: {} Score: {}"""
    weights = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )
    scores = []
    for episode in range(nb_episodes):
        state = env.reset()[0]
        gradients = []
        rewards = []
        score = 0
        done = False
        while not done:
            action, grad = policy_gradient(state, weights)
            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            gradients.append(grad)
            state = next_state
            done = terminated or truncated
        score = sum(rewards)
        scores.append(score)
        print(f"Episode: {episode} Score: {score}")
        for i, gradient in enumerate(gradients):
            cumulative_reward = 0
            for t, R in enumerate(rewards[i:]):
                cumulative_reward += R * (gamma ** t)
            weights += alpha * gradient * cumulative_reward

    return scores

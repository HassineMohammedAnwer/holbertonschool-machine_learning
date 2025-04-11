#!/usr/bin/env python3
"""3. Animate iteration"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """ implements a full training.
    env: initial environment
    nb_episodes: number of episodes used for training
    alpha: the learning rate
    gamma: the discount factor
    You should use policy_gradient =
    __import__('policy_gradient').policy_gradient
    Return: all values of the score
    __(sum of all rewards during one episode loop)
    You need print the current episode number and the score after each loop
    in a format: Episode: {} Score: {}
    _________________________________
    adding a last optional parameter show_result (default: False).
    When this parameter is set to True, you should render the
    __environment every 1000 episodes computed."""
    weight = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )
    scores = []
    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []
        done = False
        render_episode = show_result and (episode % 1000 == 0)
        while not done:
            if render_episode:
                env.render()
            action, grad = policy_gradient(state, weight)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            grads.append(grad)
            rewards.append(reward)
        total_reward = sum(rewards)
        scores.append(total_reward)
        for i in range(len(grads)):
            discounted_return = 0
            for t, r in enumerate(rewards[i:]):
                discounted_return += r * (gamma ** t)
            weight += alpha * grads[i] * discounted_return
        print(f'Episode: {episode} Score: {total_reward}')
    return scores

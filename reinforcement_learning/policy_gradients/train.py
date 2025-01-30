#!/usr/bin/env python3
"""2. Implement the training"""
import numpy as np


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
    
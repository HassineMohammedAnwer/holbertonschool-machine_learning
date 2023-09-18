#!/usr/bin/env python3
"""Defines deep NeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork class"""

    def __init__(self, nx, nodes):
        """DeepNeuralNetwork class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        for layer in layers:
            if not isinstance(layer, int) or layer <= 0:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                n = nx
            else:
                n = layers[i-1]
            m = layers[i]

            std = np.sqrt(2 / n)
            self.__weights['b' + str(i + 1)] = np.zeros((m, 1))
            self.__weights['W' + str(i + 1)] = np.random.randn(m, n) * std

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

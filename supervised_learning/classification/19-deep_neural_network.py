#!/usr/bin/env python3
""" Defines deep NeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork class"""

    def __init__(self, nx, layers):
        """DeepNeuralNetwork class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {
            'W1': np.random.randn(layers[0], nx) * np.sqrt(2 / nx),
            'b1': np.zeros((layers[0], 1))
        }

        for i in range(1, self.L):
            self.__weights["W" + str(i + 1)] =\
                np.random.randn(layers[i], layers[i - 1]) *\
                np.sqrt(2 / layers[i - 1])
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """  number of layers in the neural network"""
        return self.__L

    @property
    def weights(self):
        """ A dictionary to hold all intermediary values of the network """
        return self.__weights

    @property
    def cache(self):
        """A dictionary to hold all weights and biased of the network """
        return self.__cache

    def forward_prop(self, X):
        """forward propagation of the neural network"""
        self.__cache["A0"] = X
        A = X
        for i in range(1, self.L + 1):
            Z = np.dot(self.weights[f"W{i}"], A) + self.weights[f"b{i}"]
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f"A{i}"] = A

        return A, self.__cache

    def cost(self, Y, A):
        """cost using logistic regression"""
        return (-1 / Y.shape[1]) *\
            np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
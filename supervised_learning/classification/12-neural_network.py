#!/usr/bin/env python3
"""Defines NeuralNetwork """
import numpy as np


class NeuralNetwork:
    """NeuralNetwork class"""

    def __init__(self, nx, nodes):
        """NeuralNetwork class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """result of the first layer's activation is stored in A1,
        and the output layer's activation is stored in A2"""
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        epsilon = 0.0000001
        log_loss = -(Y * np.log(A) + (1 - Y) * np.log(epsilon + 1 - A))
        cost = np.mean(log_loss)
        return cost

    def evaluate(self, X, Y):
        """
        evaluates the Neuron's predictions.Returns prediction and the cost
        """
        A = self.forward_prop(X)[1]
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return (predictions, cost)

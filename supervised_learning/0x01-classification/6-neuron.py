#!/usr/bin/env python3
"""Neuron"""


import numpy as np


class Neuron:
    """defines a single neuron performing binary classification"""

    def __init__(self, nx):
        if not type(nx) is int:
            raise TypeError('nx must be an integer')
        elif (nx < 1):
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """Add Sigmoid Forward Prop Method"""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """defines a single"""
        m = Y.shape[1]
        xmp = (1 - Y)
        RES = - (1 / m) * (np.sum(Y * np.log(A) + xmp * np.log(1.0000001 - A)))
        return RES

    def evaluate(self, X, Y):
        """evaluates the neuron’s predictions"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        predicted_labels = np.where(self.__A >= 0.5, 1, 0)
        return predicted_labels, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """method calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dz = A - Y
        dw = np.matmul(dz, X.T) / m
        db = np.sum(dz) / m
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """checks the iterations and alpha are of the correct types and values.
        Then, it loops over the range of iterations and performs
        forward propagation and gradient descent at each iteration.
        After the loop, it evaluates the trained model on the training data
        using the evaluate method and returns the predicted values and cost."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        A, cost = self.evaluate(X, Y)
        return A, cost

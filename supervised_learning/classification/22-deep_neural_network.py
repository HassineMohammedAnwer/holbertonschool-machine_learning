#!/usr/bin/env python3
"""Defines deep NeuralNetwork """
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
        self.__weights = {'W1':
                          np.random.randn(layers[0], nx) * np.sqrt(2 / nx),
                          'b1': np.zeros((layers[0], 1))
                          }
        if not isinstance(layers[0], int) or layers[0] <= 0:
            raise TypeError("layers must be a list of positive integers")

        for i in range(1, self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W" + str(i + 1)] =\
                np.random.randn(layers[i], layers[i - 1]) *\
                np.sqrt(2 / layers[i - 1])
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ number of layers in the neural network"""
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

    def evaluate(self, X, Y):
        """Return: Returns the neuron’s prediction and the cost
           prediction: a numpy.ndarray with shape (1, m) of predicted labels
           for each example predicted labels for each example and the label
           values should be 1
           The label values should be 1 if the output of the network is >= 0.5
           and 0 otherwise
        """
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """method calculates one pass of gradient descent on the neuron
        # Number of examples
        m = Y.shape[1]
        # Derivative of the loss with respect to Z_L
        dz = A - Y
        dw = np.matmul(dz, X.T) / m
        db = np.sum(dz) / m
        # Update the weights and biases
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)"""
        m = Y.shape[1]
        L = self.L

        # Initialize gradient of the last layer (output layer)
        A_L = cache['A' + str(L)]
        dZ_L = A_L - Y
        # Backpropagation through the layers, Loop backwards through layers
        for lay in reversed(range(1, L + 1)):
            A_prev = cache['A' + str(lay - 1)]
            # Compute gradients of weights, biases, and previous activation
            dW = (1 / m) * np.matmul(dZ_L, A_prev.T)
            db = (1 / m) * np.sum(dZ_L, axis=1, keepdims=True)

            W_curr = self.weights['W' + str(lay)]
            # Sigmoid drv1
            dZ_L = np.matmul(W_curr.T, dZ_L) * (A_prev * (1 - A_prev))

            self.weights['W' + str(lay)] -= alpha * dW
            self.weights['b' + str(lay)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        A, cost = self.evaluate(X, Y)
        return A, cost"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        i = 0
        while (i < iterations):
            # run forward propagation
            self.forward_prop(X)
            # run gradient descent
            self.gradient_descent(Y, self.cache, alpha)
            i += 1

        return self.evaluate(X, Y)

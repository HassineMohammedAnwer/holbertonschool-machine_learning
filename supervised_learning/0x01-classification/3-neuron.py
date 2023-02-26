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
        cost = - (1 / m) * (np.sum(Y * np.log(A) + xmp * np.log(1.0000001 - A)))
        return cost

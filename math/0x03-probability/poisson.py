#!/usr/bin/env python3
"""Piosson module"""


class Poisson:
    """function poiss
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """def Initializer"""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number
        k is the number of “successes”"""
        if not isinstance(k, int):
            k = int(k)
        if k <= 0:
            return 0
        lpowk = self.lambtha**k
        fact = 1
        for i in range(1, k + 1):
            fact = fact * i
        return ((Poisson.e**(-1 * self.lambtha) * lpowk) / fact)

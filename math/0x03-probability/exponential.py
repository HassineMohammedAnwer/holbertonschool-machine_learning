#!/usr/bin/env python3
"""exponential class"""


class Exponential:
    """Exponential distribution class"""
    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""

        self.e = 2.7182818285
        self.lambtha = float(lambtha)

        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))

        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""
        if x < 0:
            return 0
        return self.lambtha * pow(2.7182818285, -1 * self.lambtha * x)

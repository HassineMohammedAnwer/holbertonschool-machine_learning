#!/usr/bin/env python3
""" class Normal that represents a normal distribution"""


class Normal:
    """class normal"""
    e = 2.7182818285
    pi = 3.1415926536
    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal"""
        if data is None:
            self.mean = float(mean)
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            su = 0
            for item in data:
                su += (item - self.mean) ** 2
            self.stddev = float((float(su) / len(data)) ** 0.5)

    def z_score(self, x):
        """calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """calculates the x-value of a given z-score"""
        return self.mean + z * self.stddev

    def pdf(self, x):
        """calculates the probability density function for a given x-value"""
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        coefficient = 1 / (self.stddev * ((2 * Normal.pi) ** 0.5))
        return coefficient * (Normal.e ** exponent)

#!/usr/bin/env python3
""" class Binomial that represents a binomial distribution"""


class Binomial:
    """class cinomial"""
    def __init__(self, data=None, n=1, p=0.5):
        """Initialize binomial"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            self.n = int(round(mean ** 2 / (mean - variance)))
            self.p = mean / self.n
        self.n = int(self.n)
        self.p = float(self.p)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        dnmnfct = self.factorial(self.n)
        xmp = (dnmnfct / (self.factorial(k) * self.factorial(self.n - k)))
        ymp = (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return (xmp * ymp)

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes"""
        k = int(k)
        if k < 0:
            return 0
        if k >= self.n:
            return 1
        return sum(self.pmf(i) for i in range(k + 1))

    def factorial(self, n):
        """Calculates the factorial of a given number"""
        if n == 0:
            return 1
        return n * self.factorial(n - 1)
